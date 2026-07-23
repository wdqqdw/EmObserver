"""
MiniCPM-V inference wrapper (transformers + multi-GPU DATA parallelism, 1 GPU per replica).

Models:
    ${MVEI_MODEL_ROOT}/MiniCPM-V-2_6   (~8B Qwen2 base)
    ${MVEI_MODEL_ROOT}/MiniCPM-V-4_5   (~8B Qwen2.5 base, w/ 3D-Resampler)

Strategy
--------
* Each visible GPU loads its OWN full bf16 replica of the model (pure DP, no TP).
* Input dict is round-robin sharded across workers; a shared tqdm bar
  aggregates progress in the parent process.
* MiniCPM-V is loaded via `AutoModel.from_pretrained(..., trust_remote_code=True)`
  and called through its native `model.chat(image, msgs, tokenizer, ...)`
  interface (recommended by the official model card).

Usage (identical interface to internvl3 / glm4v):
    model = MiniCPMVInferencer(model_size="4_5")
    out   = model.inference(input_dict)
    out   = model.inference(input_dict, gpu_n=8,
                            sampling_params=SimpleNamespace(n=1, temperature=0.7,
                                                            max_tokens=1024))

Input  : {id: {"image_path": str | List[str], "prompt": str, ...}}
Output : {id: {..., "generations": List[str]}}     # length == sampling_params.n
"""

import copy
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from evaluate.infer_engines.paths import model_path


MODEL_PATHS: Dict[str, str] = {
    "2_6": model_path("MiniCPM-V-2_6"),
    "4_5": model_path("MiniCPM-V-4_5"),
}
SUPPORTED_SIZES: List[str] = list(MODEL_PATHS.keys())

# 1 GPU per replica (pure DP). MiniCPM-V ~8B bf16 fits a single 24G card.
TP_PER_REPLICA: Dict[str, int] = {"2_6": 1, "4_5": 1}

# Default precision. Override via env MINICPMV_QUANT=bf16|int8|int4.
DEFAULT_QUANT: Dict[str, str] = {"2_6": "bf16", "4_5": "bf16"}


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
        from transformers import AutoModel, AutoTokenizer

        dtype = _torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        load_kwargs = dict(
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if quant == "int8":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = {"": 0}
        elif quant == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = {"": 0}
        # bf16: load to CPU then .cuda() (avoids accelerate dispatch).

        model = AutoModel.from_pretrained(model_path, **load_kwargs).eval()
        if quant == "bf16":
            model = model.cuda()

        def _normalize_paths(image_path) -> List[str]:
            if isinstance(image_path, str):
                return [image_path]
            if isinstance(image_path, (list, tuple)) and image_path:
                return list(image_path)
            raise ValueError(f"Invalid image_path: {image_path!r}")

        def _build_msgs(prompt: str, images: List["Image.Image"]) -> List[dict]:
            """
            MiniCPM-V `model.chat` accepts msgs in two forms:
              - single image  : msgs = [{"role":"user","content":[img, prompt]}]
              - multi  image  : msgs = [{"role":"user","content":[img1,img2,...,prompt]}]
            Both 2.6 and 4.5 support a list mixing PIL images + str text.
            """
            return [{"role": "user", "content": [*images, prompt]}]

        # Generation knobs accepted by MiniCPM-V's chat method
        gen_cfg = dict(
            max_new_tokens=int(gen_kwargs.get("max_new_tokens", 1024)),
            do_sample=bool(gen_kwargs.get("do_sample", False)),
            temperature=float(gen_kwargs.get("temperature", 1.0)),
            top_p=float(gen_kwargs.get("top_p", 1.0)),
        )

        local_result: Dict[str, Dict] = {}
        for sid, sample in shard.items():
            try:
                paths = _normalize_paths(sample["image_path"])
                images = [Image.open(p).convert("RGB") for p in paths]
                msgs = _build_msgs(sample["prompt"], images)

                with _torch.inference_mode():
                    # MiniCPM-V official call:
                    #   res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, ...)
                    # `image=None` because images are already embedded in msgs.
                    response = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=gen_cfg["do_sample"],
                        temperature=gen_cfg["temperature"],
                        top_p=gen_cfg["top_p"],
                        max_new_tokens=gen_cfg["max_new_tokens"],
                    )

                if isinstance(response, tuple):
                    response = response[0]
                out_text = response.strip() if isinstance(response, str) else str(response)

            except Exception as e:
                out_text = f"[MiniCPMV-ERROR] {type(e).__name__}: {e}"

            rec = copy.deepcopy(sample)
            rec["generations"] = [out_text] * n_gen
            local_result[sid] = rec
            progress_q.put(1)

        result_q.put((worker_id, local_result))

    except Exception as e:
        import traceback
        result_q.put((
            worker_id,
            {"__error__": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"},
        ))


# =============================================================================
# Public API
# =============================================================================
class MiniCPMVInferencer:
    def __init__(self, model_size: str = "4_5", quant: Optional[str] = None):
        key = model_size.lower()
        if key not in MODEL_PATHS:
            raise ValueError(
                f"Unknown model_size={model_size!r}. Supported: {SUPPORTED_SIZES}"
            )
        self.model_size = key
        self.model_path = MODEL_PATHS[key]
        self.quant = (quant
                      or os.environ.get("MINICPMV_QUANT")
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

        print(f"[MiniCPM-V] size={self.model_size} quant={self.quant} "
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
        desc = f"MiniCPM-V-{self.model_size}[{self.quant}] DP{n_replicas}"
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
            raise RuntimeError("MiniCPM-V worker(s) crashed:\n" + "\n".join(errors))
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
        m = MiniCPMVInferencer(model_size="4_5")  # bf16 by default
        res = m.inference(small, gpu_n=8)
        print(json.dumps(res, ensure_ascii=False, indent=2))