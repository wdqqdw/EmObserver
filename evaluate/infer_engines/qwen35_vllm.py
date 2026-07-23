"""
Qwen3.5 (multimodal MoE) inference wrapper backed by vLLM.
"""

import copy
from typing import Dict, List, Optional, Union


# ===========================================================================
# Hot-patch: neutralize ONLY `_check_received_keys` on
# RotaryEmbeddingConfigMixin. Do NOT patch `validate_rope` — huggingface_hub's
# @strict decorator (used on PreTrainedConfig) requires methods starting with
# `validate_` to take only `self`.
# ===========================================================================
def _disable_rope_validation():
    import transformers.modeling_rope_utils as _mru

    cls = getattr(_mru, "RotaryEmbeddingConfigMixin", None)
    if cls is None:
        return

    def _noop_check(self, *args, **kwargs):
        return None
    _noop_check._qwen35_patched = True

    if "_check_received_keys" in cls.__dict__:
        cls._check_received_keys = _noop_check


_disable_rope_validation()


import torch  # noqa: E402
from PIL import Image  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from evaluate.infer_engines.paths import model_path


MODEL_PATHS: Dict[str, str] = {
    "9b":  model_path("Qwen3.5-9B"),
    "27b": model_path("Qwen3.5-27B"),
}
SUPPORTED_SIZES: List[str] = list(MODEL_PATHS.keys())


class Qwen35Inferencer:
    def __init__(self, model_size: str = "9b"):
        key = model_size.lower()
        if key not in MODEL_PATHS:
            raise ValueError(
                f"Unknown model_size={model_size!r}. Supported: {SUPPORTED_SIZES}"
            )
        self.model_size = key
        self.model_path = MODEL_PATHS[key]

        self._llm: Optional[LLM] = None
        self._llm_gpu_n: Optional[int] = None

    @staticmethod
    def _resolve_gpu_n(gpu_n: Optional[int]) -> int:
        if gpu_n is not None:
            return gpu_n
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        raise RuntimeError("No CUDA devices visible to the process.")

    def _ensure_llm(self, gpu_n: int):
        if self._llm is not None and self._llm_gpu_n == gpu_n:
            return
        if self._llm is not None:
            del self._llm
            torch.cuda.empty_cache()

        kwargs = dict(
            model=self.model_path,
            tensor_parallel_size=gpu_n,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 8},
            dtype="bfloat16",
            enable_prefix_caching=True,
            reasoning_parser="qwen3",
            mm_encoder_tp_mode="data",
            mm_processor_cache_type="shm",
        )
        try:
            self._llm = LLM(**kwargs)
        except TypeError:
            for k in ("mm_encoder_tp_mode", "mm_processor_cache_type",
                      "reasoning_parser", "enable_prefix_caching"):
                kwargs.pop(k, None)
            self._llm = LLM(**kwargs)

        self._llm_gpu_n = gpu_n

    @staticmethod
    def _load_images(image_path: Union[str, List[str]]) -> List[Image.Image]:
        paths = [image_path] if isinstance(image_path, str) else list(image_path)
        return [Image.open(p).convert("RGB") for p in paths]

    @staticmethod
    def _build_messages(prompt: str, n_images: int) -> List[dict]:
        content = [{"type": "image"} for _ in range(n_images)]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def inference(
        self,
        input: Dict[str, Dict],
        gpu_n: Optional[int] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Dict[str, Dict]:
        gpu_n = self._resolve_gpu_n(gpu_n)
        self._ensure_llm(gpu_n)

        if sampling_params is None:
            sampling_params = SamplingParams(n=1, temperature=0.7, max_tokens=1024)

        from transformers import AutoProcessor
        if not hasattr(self, "_processor"):
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )

        ids: List[str] = list(input.keys())
        vllm_inputs = []
        for sid in ids:
            sample = input[sid]
            images = self._load_images(sample["image_path"])
            messages = self._build_messages(sample["prompt"], len(images))
            text_prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            vllm_inputs.append({
                "prompt": text_prompt,
                "multi_modal_data": {"image": images},
            })

        outputs = self._llm.generate(vllm_inputs, sampling_params=sampling_params)

        result: Dict[str, Dict] = {}
        for sid, out in zip(ids, outputs):
            generations = [o.text for o in out.outputs]
            new_record = copy.deepcopy(input[sid])
            new_record["generations"] = generations
            result[sid] = new_record
        return result


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) >= 2:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            data = json.load(f)
        small = dict(list(data.items())[:4])
        m = Qwen35Inferencer(model_size="9b")
        res = m.inference(small)
        print(json.dumps(res, ensure_ascii=False, indent=2))