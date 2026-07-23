"""
GLM-4.x V inference wrapper (transformers + multi-GPU DATA parallelism, 1 GPU per replica).

Aligned strictly with the official GLM-4.6V-Flash reference code.
"""

import copy
import os
import sys
import traceback as _tb
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from evaluate.infer_engines.paths import model_path


MODEL_PATHS: Dict[str, str] = {
    "4_1v_thinking": model_path("GLM-4.1V-9B-Thinking"),
    "4_6v_flash":    model_path("GLM-4.6V-Flash"),
}
SUPPORTED_SIZES: List[str] = list(MODEL_PATHS.keys())

TP_PER_REPLICA: Dict[str, int] = {"4_1v_thinking": 1, "4_6v_flash": 1}
DEFAULT_QUANT: Dict[str, str] = {"4_1v_thinking": "bf16", "4_6v_flash": "bf16"}


# =============================================================================
# transformers.video_utils.VideoMetadata kwargs hot-patch
# =============================================================================
def _patch_video_metadata_kwargs():
    try:
        from transformers.video_utils import VideoMetadata
        import inspect
        if getattr(VideoMetadata, "_patched_drop_unknown", False):
            return
        sig = inspect.signature(VideoMetadata.__init__)
        known = set(sig.parameters.keys()) - {"self"}
        orig_init = VideoMetadata.__init__

        def _init(self, *args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k in known}
            orig_init(self, *args, **kwargs)

        VideoMetadata.__init__ = _init
        VideoMetadata._patched_drop_unknown = True
    except Exception as e:
        print(f"[GLM4V-WARN] patch_video_metadata_kwargs failed: {e}")


# =============================================================================
# Worker
# =============================================================================
def _worker_main(
    worker_id: int,
    gpu_id: int,
    model_path: str,
    quant: str,
    shard: Dict[str, Dict],
    n_gen: int,
    gen_kwargs: dict,
    progress_q: "mp.Queue",
    result_q: "mp.Queue",
):
    # Pin BEFORE importing torch CUDA APIs.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        import torch as _torch
        from PIL import Image
        from transformers import AutoProcessor, AutoTokenizer

        _patch_video_metadata_kwargs()

        try:
            from transformers import Glm4vForConditionalGeneration as _ModelCls
        except ImportError:
            from transformers import AutoModelForCausalLM as _ModelCls

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Load EXACTLY as official: torch_dtype="auto" + device_map="auto".
        # Since CUDA_VISIBLE_DEVICES is pinned to a single GPU above, device_map="auto"
        # places everything on that one card.
        load_kwargs = dict(
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )

        if quant == "int8":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["torch_dtype"] = _torch.bfloat16
        elif quant == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["torch_dtype"] = _torch.bfloat16

        model = _ModelCls.from_pretrained(model_path, **load_kwargs).eval()

        def _normalize_paths(image_path) -> List[str]:
            if isinstance(image_path, str):
                return [image_path]
            if isinstance(image_path, (list, tuple)) and image_path:
                return list(image_path)
            raise ValueError(f"Invalid image_path: {image_path!r}")

        def _build_messages(prompt: str, images: List["Image.Image"]) -> List[dict]:
            content: List[dict] = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt})
            return [{"role": "user", "content": content}]

        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 1024))
        do_sample = bool(gen_kwargs.get("do_sample", False))
        temperature = float(gen_kwargs.get("temperature", 1.0))
        top_p = float(gen_kwargs.get("top_p", 1.0))

        local_result: Dict[str, Dict] = {}
        for sid, sample in shard.items():
            try:
                paths = _normalize_paths(sample["image_path"])
                images = [Image.open(p).convert("RGB") for p in paths]
                messages = _build_messages(sample["prompt"], images)

                # === STRICT alignment with official reference ===
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                # Official does exactly this single pop:
                inputs.pop("token_type_ids", None)

                input_len = inputs["input_ids"].shape[1]

                # Match official: only pass max_new_tokens.
                # Add sampling controls ONLY when explicitly enabled.
                gen_extra = {"max_new_tokens": max_new_tokens}
                if do_sample:
                    gen_extra["do_sample"] = True
                    gen_extra["temperature"] = temperature
                    gen_extra["top_p"] = top_p

                with _torch.inference_mode():
                    generated_ids = model.generate(**inputs, **gen_extra)

                if generated_ids is None:
                    raise RuntimeError("model.generate returned None")

                # Decode via processor (official style)
                gen_ids = generated_ids[0][input_len:]
                out_text = processor.decode(gen_ids, skip_special_tokens=True).strip()

            except Exception as e:
                out_text = f"[GLM4V-ERROR] {type(e).__name__}: {e} | trace={_tb.format_exc(limit=4)}"

            rec = copy.deepcopy(sample)
            rec["generations"] = [out_text] * n_gen
            local_result[sid] = rec
            progress_q.put(1)

        result_q.put((worker_id, local_result))

    except Exception as e:
        result_q.put((
            worker_id,
            {"__error__": f"{type(e).__name__}: {e}\n{_tb.format_exc()}"},
        ))


# =============================================================================
# Public API (unchanged)
# =============================================================================
class GLM46VInferencer:
    def __init__(self, model_size: str = "4_1v_thinking", quant: Optional[str] = None):
        key = model_size.lower()
        if key not in MODEL_PATHS:
            raise ValueError(
                f"Unknown model_size={model_size!r}. Supported: {SUPPORTED_SIZES}"
            )
        self.model_size = key
        self.model_path = MODEL_PATHS[key]
        self.quant = (quant
                      or os.environ.get("GLM4V_QUANT")
                      or DEFAULT_QUANT[key]).lower()
        if self.quant not in ("bf16", "int8", "int4"):
            raise ValueError(f"quant must be bf16/int8/int4, got {self.quant!r}")

    @staticmethod
    def _resolve_gpu_n(gpu_n: Optional[int]) -> int:
        if gpu_n is not None:
            return gpu_n
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        raise RuntimeError("No CUDA devices visible to the process.")

    @staticmethod
    def _shard(input_dict: Dict[str, Dict], k: int) -> List[Dict[str, Dict]]:
        shards = [dict() for _ in range(k)]
        for i, (sid, sample) in enumerate(input_dict.items()):
            shards[i % k][sid] = sample
        return shards

    @staticmethod
    def _gen_kwargs_from_sp(sp) -> dict:
        if sp is None:
            return {"max_new_tokens": 1024, "do_sample": False,
                    "temperature": 1.0, "top_p": 1.0}
        temperature = float(getattr(sp, "temperature", 1.0) or 1.0)
        top_p       = float(getattr(sp, "top_p", 1.0) or 1.0)
        max_tokens  = int(getattr(sp, "max_tokens",
                                  getattr(sp, "max_new_tokens", 1024)) or 1024)
        do_sample   = temperature > 0 and temperature != 1.0
        return {
            "max_new_tokens": max_tokens,
            "do_sample":      do_sample,
            "temperature":    max(temperature, 1e-5),
            "top_p":          top_p,
        }

    def inference(
        self,
        input: Dict[str, Dict],
        gpu_n: Optional[int] = None,
        sampling_params=None,
    ) -> Dict[str, Dict]:
        if not input:
            return {}

        gpu_n = self._resolve_gpu_n(gpu_n)
        n_gen = int(getattr(sampling_params, "n", 1) or 1)
        gen_kwargs = self._gen_kwargs_from_sp(sampling_params)

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            phys_ids = [int(x) for x in visible.split(",") if x.strip() != ""]
            if len(phys_ids) < gpu_n:
                raise RuntimeError(
                    f"gpu_n={gpu_n} but CUDA_VISIBLE_DEVICES only exposes {phys_ids}")
            phys_ids = phys_ids[:gpu_n]
        else:
            phys_ids = list(range(gpu_n))

        n_replicas = len(phys_ids)
        shards = self._shard(input, n_replicas)

        ctx = mp.get_context("spawn")
        progress_q = ctx.Queue()
        result_q   = ctx.Queue()

        print(f"[GLM4V] size={self.model_size} quant={self.quant} "
              f"DP={n_replicas} (1 GPU per replica)  samples={len(input)}")
        for wid, (gid, sh) in enumerate(zip(phys_ids, shards)):
            print(f"  replica {wid}: GPU={gid}  shard_size={len(sh)}")

        procs = []
        for worker_id, (gpu_id, shard) in enumerate(zip(phys_ids, shards)):
            p = ctx.Process(
                target=_worker_main,
                args=(worker_id, gpu_id, self.model_path, self.quant, shard,
                      n_gen, gen_kwargs, progress_q, result_q),
                daemon=False,
            )
            p.start()
            procs.append(p)

        total = len(input)
        merged: Dict[str, Dict] = {}
        errors: List[str] = []
        finished_workers = 0
        desc = f"GLM4V-{self.model_size}[{self.quant}] DP{n_replicas}"
        with tqdm(total=total, desc=desc, dynamic_ncols=True) as pbar:
            while finished_workers < len(procs) or not result_q.empty() or not progress_q.empty():
                drained = 0
                while True:
                    try:
                        progress_q.get_nowait()
                        drained += 1
                    except Exception:
                        break
                if drained:
                    pbar.update(drained)

                try:
                    worker_id, local = result_q.get(timeout=0.2)
                    if "__error__" in local:
                        errors.append(f"[worker {worker_id}] {local['__error__']}")
                    else:
                        merged.update(local)
                    finished_workers += 1
                except Exception:
                    pass

                if all(not p.is_alive() for p in procs) and result_q.empty() and progress_q.empty():
                    break

        for p in procs:
            p.join()

        if errors:
            raise RuntimeError("GLM4V worker(s) crashed:\n" + "\n".join(errors))
        if len(merged) != total:
            missing = set(input.keys()) - set(merged.keys())
            raise RuntimeError(f"Missing {len(missing)} results, e.g. {list(missing)[:5]}")

        return {sid: merged[sid] for sid in input.keys()}


if __name__ == "__main__":
    import json
    if len(sys.argv) >= 2:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            data = json.load(f)
        small = dict(list(data.items())[:8])
        m = GLM46VInferencer(model_size="4_6v_flash")
        res = m.inference(small, gpu_n=8)
        print(json.dumps(res, ensure_ascii=False, indent=2))