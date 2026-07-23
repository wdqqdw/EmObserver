"""
Qwen3-VL inference wrapper backed by vLLM.

Two construction modes — both share the SAME inference() pipeline:

(A) Registered mode  (used by evaluate/infer_and_eval.py):
        model = Qwen3VLInferencer(model_size="8b_instruct")
    -> Resolves model path via MODEL_PATHS[model_size].

(B) Checkpoint mode  (used by evaluate/infer_and_eval_qwen3vl.py):
        model = Qwen3VLInferencer(model_path="/path/to/your/finetuned/ckpt")
    -> Bypasses MODEL_PATHS entirely. Optional `is_thinking=True/False`.

Usage:
    model = Qwen3VLInferencer(model_size="8b_instruct")
    out = model.inference(input_dict)            # uses all visible GPUs, n=1, temp=0.7
    out = model.inference(input_dict, gpu_n=4, sampling_params=SamplingParams(n=4, temperature=1.0))

Input  : {id: {"image_path": str | List[str], "prompt": str, ...}}
Output : {id: {..., "generations": List[str]}}   # length == sampling_params.n
"""

import copy
import os
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from vllm import LLM, SamplingParams
from evaluate.infer_engines.paths import model_path


# All supported size tags. Keys are case-insensitive on the user side.
MODEL_PATHS: Dict[str, str] = {
    "8b_instruct":  model_path("Qwen3-VL-8B-Instruct"),
    "8b_think":     model_path("Qwen3-VL-8B-Thinking"),
    "32b_instruct": model_path("Qwen3-VL-32B-Instruct"),
    "32b_think":    model_path("Qwen3-VL-32B-Thinking"),
    "emobserver":  model_path("EmObserver"),
    # Extend here when more variants are available, e.g.:
    # "4b_think": "...",
    # "4b_instruct": "...",
}

SUPPORTED_SIZES: List[str] = list(MODEL_PATHS.keys())

# Heuristic tags for auto-detecting thinking variants from a path/size string.
_THINK_TAGS = ("think", "thinking", "reason", "emobserver")


class Qwen3VLInferencer:
    def __init__(
        self,
        model_size: Optional[str] = "8b_instruct",
        model_path: Optional[str] = None,
        size: Optional[str] = None,              # alias accepted for safety
        is_thinking: Optional[bool] = None,
        **_unused,
    ):
        # Allow `size=` as an alias for `model_size=`.
        if size is not None and model_size == "8b_instruct":
            model_size = size

        # ---- Resolve model path ------------------------------------------------
        if model_path:
            # Checkpoint mode: skip MODEL_PATHS lookup entirely.
            if not os.path.isdir(model_path):
                raise FileNotFoundError(
                    f"Qwen3VLInferencer: model_path is not a directory: {model_path}"
                )
            self.model_path = model_path
            # Keep a human-readable tag for logging / progress bars.
            self.model_size = (
                model_size or os.path.basename(model_path.rstrip("/")) or "ckpt"
            ).lower()
            tag_for_thinking = (model_size or model_path).lower()
        else:
            if model_size is None:
                raise ValueError(
                    "Qwen3VLInferencer requires either `model_path` or `model_size`."
                )
            key = model_size.lower()
            if key not in MODEL_PATHS:
                raise ValueError(
                    f"Unknown model_size={model_size!r}. "
                    f"Supported: {SUPPORTED_SIZES}"
                )
            self.model_size = key
            self.model_path = MODEL_PATHS[key]
            tag_for_thinking = key

        # ---- Decide thinking-mode (controls reasoning_parser) ------------------
        if is_thinking is None:
            is_thinking = any(t in tag_for_thinking for t in _THINK_TAGS)
        self.is_thinking: bool = bool(is_thinking)

        # Defer LLM instantiation to first inference call so we can honor `gpu_n`
        # at runtime. (vLLM's tensor_parallel_size is fixed at LLM construction.)
        self._llm: Optional[LLM] = None
        self._llm_gpu_n: Optional[int] = None

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _resolve_gpu_n(gpu_n: Optional[int]) -> int:
        if gpu_n is not None:
            return gpu_n
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        raise RuntimeError("No CUDA devices visible to the process.")

    def _ensure_llm(self, gpu_n: int):
        """Instantiate (or re-instantiate if gpu_n changed) the underlying vLLM engine."""
        if self._llm is not None and self._llm_gpu_n == gpu_n:
            return
        if self._llm is not None:
            del self._llm
            torch.cuda.empty_cache()

        kwargs = dict(
            model=self.model_path,
            tensor_parallel_size=gpu_n,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 8},   # allow multi-image prompts
            dtype="bfloat16",
        )
        # Enable Qwen3 reasoning parser ONLY for thinking variants.
        if self.is_thinking:
            kwargs["reasoning_parser"] = "qwen3"
        else:
            # Original behavior: explicitly None for non-thinking variants.
            kwargs["reasoning_parser"] = None

        try:
            self._llm = LLM(**kwargs)
        except TypeError:
            # Some vLLM builds may not accept `reasoning_parser`; drop it and retry.
            kwargs.pop("reasoning_parser", None)
            self._llm = LLM(**kwargs)

        self._llm_gpu_n = gpu_n

    @staticmethod
    def _load_images(image_path: Union[str, List[str]]) -> List[Image.Image]:
        paths = [image_path] if isinstance(image_path, str) else list(image_path)
        return [Image.open(p).convert("RGB") for p in paths]

    @staticmethod
    def _build_messages(prompt: str, n_images: int) -> List[dict]:
        """Qwen3-VL chat template: a single user turn with N image segments + text."""
        content = [{"type": "image"} for _ in range(n_images)]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    # ---------------------------------------------------------------- public
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
    # Minimal smoke test.
    import json, sys
    if len(sys.argv) >= 2:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            data = json.load(f)
        small = dict(list(data.items())[:4])
        m = Qwen3VLInferencer(model_size="8b_instruct")
        res = m.inference(small)
        print(json.dumps(res, ensure_ascii=False, indent=2))