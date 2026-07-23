"""
BLIP-2 inference wrapper (transformers + multi-GPU data parallelism).

Strategy
--------
* Each visible GPU loads its own full copy of the model. Input dict is
  round-robin sharded across workers; a shared tqdm bar aggregates progress.
* BLIP-2 is single-image — for multi-image samples we feed the FIRST image.
* Prompt format follows BLIP-2 VQA convention: "Question: {q} Answer:".

Usage (identical interface to qwen3_vl_vllm.py):
    model = BLIP2Inferencer(model_size="opt_6_7b")
    out   = model.inference(input_dict)
    out   = model.inference(input_dict, gpu_n=8,
                            sampling_params=SimpleNamespace(n=1, temperature=0.7,
                                                            max_tokens=256))

Input  : {id: {"image_path": str | List[str], "prompt": str, ...}}
Output : {id: {..., "generations": List[str]}}     # length == sampling_params.n
"""

import copy
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from evaluate.infer_engines.paths import model_path


MODEL_PATHS: Dict[str, str] = {
    "opt_6_7b": model_path("blip2-opt-6.7b"),
}
SUPPORTED_SIZES: List[str] = list(MODEL_PATHS.keys())


# =============================================================================
# Worker (runs in a child process; pinned to ONE physical GPU)
# =============================================================================
def _worker_main(
    gpu_id: int,
    model_path: str,
    shard: Dict[str, Dict],
    n_gen: int,
    gen_kwargs: dict,
    progress_q: "mp.Queue",
    result_q: "mp.Queue",
):
    # Pin BEFORE importing torch-using libs in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        import torch as _torch
        from PIL import Image
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        device = _torch.device("cuda:0")  # only one visible -> always cuda:0
        dtype = _torch.float16            # BLIP-2-OPT-6.7B was released in fp16

        processor = Blip2Processor.from_pretrained(model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        def _first_image(image_path) -> "Image.Image":
            if isinstance(image_path, str):
                p = image_path
            elif isinstance(image_path, (list, tuple)) and image_path:
                p = image_path[0]
            else:
                raise ValueError(f"Invalid image_path: {image_path!r}")
            return Image.open(p).convert("RGB")

        def _build_prompt(q: str) -> str:
            return f"Question: {q} Answer:"

        local_result: Dict[str, Dict] = {}
        for sid, sample in shard.items():
            try:
                image = _first_image(sample["image_path"])
                prompt = _build_prompt(sample["prompt"])

                inputs = processor(
                    images=image, text=prompt, return_tensors="pt"
                ).to(device, dtype)

                with _torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=int(gen_kwargs.get("max_new_tokens", 256)),
                        do_sample=bool(gen_kwargs.get("do_sample", False)),
                        temperature=float(gen_kwargs.get("temperature", 1.0)),
                        top_p=float(gen_kwargs.get("top_p", 1.0)),
                        num_beams=int(gen_kwargs.get("num_beams", 1)),
                    )
                # BLIP-2 generate already returns ONLY the answer tokens
                # (it doesn't echo the prompt), so no slicing needed.
                output_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()

            except Exception as e:
                output_text = f"[BLIP2-ERROR] {type(e).__name__}: {e}"

            new_record = copy.deepcopy(sample)
            new_record["generations"] = [output_text] * n_gen
            local_result[sid] = new_record
            progress_q.put(1)

        result_q.put((gpu_id, local_result))

    except Exception as e:
        import traceback
        result_q.put((
            gpu_id,
            {"__error__": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"},
        ))


# =============================================================================
# Public API (parent process)
# =============================================================================
class BLIP2Inferencer:
    def __init__(self, model_size: str = "opt_6_7b"):
        key = model_size.lower()
        if key not in MODEL_PATHS:
            raise ValueError(f"Unknown model_size={model_size!r}. Supported: {SUPPORTED_SIZES}")
        self.model_size = key
        self.model_path = MODEL_PATHS[key]

    @staticmethod
    def _resolve_gpu_n(gpu_n: Optional[int]) -> int:
        if gpu_n is not None:
            return gpu_n
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        raise RuntimeError("No CUDA devices visible to the process.")

    @staticmethod
    def _shard(input_dict: Dict[str, Dict], k: int) -> List[Dict[str, Dict]]:
        shards: List[Dict[str, Dict]] = [dict() for _ in range(k)]
        for i, (sid, sample) in enumerate(input_dict.items()):
            shards[i % k][sid] = sample
        return shards

    @staticmethod
    def _gen_kwargs_from_sp(sp) -> dict:
        if sp is None:
            return {"max_new_tokens": 256, "do_sample": False,
                    "temperature": 1.0, "top_p": 1.0, "num_beams": 1}
        temperature = float(getattr(sp, "temperature", 1.0) or 1.0)
        top_p       = float(getattr(sp, "top_p", 1.0) or 1.0)
        max_tokens  = int(getattr(sp, "max_tokens",
                                  getattr(sp, "max_new_tokens", 256)) or 256)
        do_sample   = temperature > 0 and temperature != 1.0
        return {
            "max_new_tokens": max_tokens,
            "do_sample":      do_sample,
            "temperature":    max(temperature, 1e-5),
            "top_p":          top_p,
            "num_beams":      1,
        }

    def inference(
        self,
        input: Dict[str, Dict],
        gpu_n: Optional[int] = None,
        sampling_params=None,
    ) -> Dict[str, Dict]:
        gpu_n = self._resolve_gpu_n(gpu_n)
        n_gen = int(getattr(sampling_params, "n", 1) or 1)
        gen_kwargs = self._gen_kwargs_from_sp(sampling_params)

        if not input:
            return {}

        # Resolve physical GPU ids (honor outer CUDA_VISIBLE_DEVICES).
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            phys_ids = [int(x) for x in visible.split(",") if x.strip() != ""]
            if len(phys_ids) < gpu_n:
                raise RuntimeError(
                    f"gpu_n={gpu_n} but CUDA_VISIBLE_DEVICES only exposes {phys_ids}"
                )
            phys_ids = phys_ids[:gpu_n]
        else:
            phys_ids = list(range(gpu_n))

        shards = self._shard(input, gpu_n)
        ctx = mp.get_context("spawn")
        progress_q = ctx.Queue()
        result_q   = ctx.Queue()

        procs = []
        for gpu_id, shard in zip(phys_ids, shards):
            p = ctx.Process(
                target=_worker_main,
                args=(gpu_id, self.model_path, shard, n_gen, gen_kwargs,
                      progress_q, result_q),
                daemon=False,
            )
            p.start()
            procs.append(p)

        total = len(input)
        merged: Dict[str, Dict] = {}
        errors: List[str] = []
        finished_workers = 0
        with tqdm(total=total, desc=f"BLIP2 x{gpu_n}", dynamic_ncols=True) as pbar:
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
                    gpu_id, local = result_q.get(timeout=0.2)
                    if "__error__" in local:
                        errors.append(f"[gpu {gpu_id}] {local['__error__']}")
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
            raise RuntimeError("BLIP2 worker(s) crashed:\n" + "\n".join(errors))
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
        m = BLIP2Inferencer(model_size="opt_6_7b")
        res = m.inference(small, gpu_n=2)
        print(json.dumps(res, ensure_ascii=False, indent=2))