"""
MiMo-VL inference wrapper backed by vLLM.

Usage:
    model = MiMoVLInferencer(model_size="7b")
    out = model.inference(input_dict)            # uses all visible GPUs
    out = model.inference(
        input_dict,
        gpu_n=4,
        sampling_params=SamplingParams(n=4, temperature=1.0, max_tokens=1024),
    )

Input  : {id: {"image_path": str | List[str], "prompt": str, ...}}
Output : {id: {..., "generations": List[str]}}   # length == sampling_params.n

Notes
-----
* This implementation assumes MiMo-VL is compatible with vLLM's multimodal
  prompt format and the model's processor supports apply_chat_template().
* If your local MiMo-VL checkpoint uses a different prompt convention, adjust
  _build_messages() and/or the prompt rendering section accordingly.
"""

import copy
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from vllm import LLM, SamplingParams
from evaluate.infer_engines.paths import model_path


MODEL_PATHS: Dict[str, str] = {
    "7b": model_path("MiMo-VL-7B-RL"),
    # 如有其他规格可继续补：
    # "13b": model_path("MiMo-VL-13B"),
}
SUPPORTED_SIZES: List[str] = list(MODEL_PATHS.keys())


class MiMoVLInferencer:
    def __init__(self, model_size: str = "7b"):
        key = model_size.lower()
        if key not in MODEL_PATHS:
            raise ValueError(
                f"Unknown model_size={model_size!r}. Supported: {SUPPORTED_SIZES}"
            )
        self.model_size = key
        self.model_path = MODEL_PATHS[key]

        # vLLM engine is initialized lazily so tensor_parallel_size can follow gpu_n.
        self._llm: Optional[LLM] = None
        self._llm_gpu_n: Optional[int] = None
        self._processor = None

    # ------------------------------------------------------------------ utils
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

        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=gpu_n,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 8},
            dtype="bfloat16",
        )
        self._llm_gpu_n = gpu_n

    def _ensure_processor(self):
        if self._processor is not None:
            return

        from transformers import AutoProcessor
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

    @staticmethod
    def _load_images(image_path: Union[str, List[str]]) -> List[Image.Image]:
        paths = [image_path] if isinstance(image_path, str) else list(image_path)
        return [Image.open(p).convert("RGB") for p in paths]

    @staticmethod
    def _build_messages(prompt: str, n_images: int) -> List[dict]:
        """
        Generic multimodal chat format:
        one user turn with N image segments followed by one text segment.
        """
        content = [{"type": "image"} for _ in range(n_images)]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _render_prompt(self, messages: List[dict]) -> str:
        """
        Prefer processor.apply_chat_template(); fall back to a conservative
        textual prompt if the checkpoint does not expose it.
        """
        if hasattr(self._processor, "apply_chat_template"):
            try:
                return self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                # Some processors may not support add_generation_prompt
                return self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                )

        # Fallback: convert images to textual placeholders.
        user_msg = messages[0]["content"]
        parts = []
        for item in user_msg:
            if item["type"] == "image":
                parts.append("<image>")
            elif item["type"] == "text":
                parts.append(item["text"])
        return "".join(parts)

    # ---------------------------------------------------------------- public
    def inference(
        self,
        input: Dict[str, Dict],
        gpu_n: Optional[int] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Dict[str, Dict]:
        gpu_n = self._resolve_gpu_n(gpu_n)
        self._ensure_llm(gpu_n)
        self._ensure_processor()

        if sampling_params is None:
            sampling_params = SamplingParams(
                n=1,
                temperature=0.7,
                max_tokens=1024,
            )

        ids: List[str] = list(input.keys())
        vllm_inputs = []

        for sid in ids:
            sample = input[sid]
            images = self._load_images(sample["image_path"])
            messages = self._build_messages(sample["prompt"], len(images))
            text_prompt = self._render_prompt(messages)

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
    import json
    import sys

    if len(sys.argv) >= 2:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            data = json.load(f)

        small = dict(list(data.items())[:4])
        m = MiMoVLInferencer(model_size="7b")
        res = m.inference(
            small,
            sampling_params=SamplingParams(n=1, temperature=0.0, max_tokens=512),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))