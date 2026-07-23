"""
InternVL3 inference wrapper (transformers + multi-GPU DATA parallelism).

Models:
    ${MVEI_MODEL_ROOT}/InternVL3-8B    (1 GPU / replica, bf16)
    ${MVEI_MODEL_ROOT}/InternVL3-38B   (1 GPU / replica, bf16)
    ${MVEI_MODEL_ROOT}/InternVL3-78B   (N GPUs / replica, bf16, pipeline parallel via device_map="auto")

Strategy
--------
* 8B / 38B: each visible GPU loads its OWN full bf16 replica (pure DP, no TP).
* 78B:     each replica spans GPUS_PER_REPLICA cards (default 4) using HF
           `device_map="auto"` to shard layers; replicas still run in parallel.
* Input dict is round-robin sharded across replicas; a shared tqdm bar
  aggregates progress in the parent process.
* InternVL3 uses the dynamic 448x448 tile-based image pipeline and the
  `<image>\n...` token convention, identical to InternVL3.5.

Usage (identical interface to internvl3_5.py):
    model = InternVL3Inferencer(model_size="8b")
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
    "8b":  model_path("InternVL3-8B"),
    "38b": model_path("InternVL3-38B"),
    "78b": model_path("InternVL3-78B"),
}
SUPPORTED_SIZES: List[str] = list(MODEL_PATHS.keys())

# GPUs per replica.
# - 8B / 38B: 1 card per replica (pure DP)
# - 78B    : multi-card per replica (bf16 ~150GB, needs >=2 x 80G; default 4)
#   Override with env INTERNVL3_GPUS_PER_REPLICA
DEFAULT_GPUS_PER_REPLICA: Dict[str, int] = {"8b": 1, "38b": 1, "78b": 2}

# Default precision per size. Override via env INTERNVL3_QUANT=bf16|int8|int4.
DEFAULT_QUANT: Dict[str, str] = {"8b": "bf16", "38b": "bf16", "78b": "bf16"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_PATCH_SIZE = 448
MIN_NUM_TILES  = 1
MAX_NUM_TILES  = 12


# =============================================================================
# Image preprocessing (InternVL official tile pipeline)
# =============================================================================
def _build_transform(input_size: int = IMG_PATCH_SIZE):
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from torchvision.transforms import InterpolationMode
    return Compose([
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf"); best = (1, 1); area = width * height
    for ratio in target_ratios:
        tgt = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tgt)
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best = ratio
    return best


def _dynamic_preprocess(image, min_num=MIN_NUM_TILES, max_num=MAX_NUM_TILES,
                        image_size=IMG_PATCH_SIZE, use_thumbnail=True):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
                for i in range(1, n + 1) for j in range(1, n + 1)
                if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    target_ar = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    tw, th = image_size * target_ar[0], image_size * target_ar[1]
    blocks = target_ar[0] * target_ar[1]
    resized = image.resize((tw, th))
    processed = []
    for i in range(blocks):
        box = (
            (i % (tw // image_size)) * image_size,
            (i // (tw // image_size)) * image_size,
            ((i % (tw // image_size)) + 1) * image_size,
            ((i // (tw // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _load_image_as_tiles(image_path: str, max_num: int = MAX_NUM_TILES,
                         input_size: int = IMG_PATCH_SIZE) -> torch.Tensor:
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    transform = _build_transform(input_size=input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size,
                                use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


# =============================================================================
# Apex bypass (with valid __spec__, installed AFTER transformers import)
# =============================================================================
def _bypass_apex():
    import sys as _sys, types
    import importlib.machinery as _mach
    import torch as _torch
    import torch.nn as nn

    class _PyRMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, **kw):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.eps = eps
            if elementwise_affine:
                self.weight = nn.Parameter(_torch.ones(normalized_shape))
            else:
                self.register_parameter("weight", None)
        def forward(self, x):
            dtype = x.dtype
            xf = x.float()
            xf = xf * _torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            out = xf.to(dtype)
            return out * self.weight if self.weight is not None else out

    class _PyFusedLayerNorm(nn.LayerNorm):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, **kw):
            super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def _mk(name: str, is_pkg: bool = False) -> types.ModuleType:
        mod = types.ModuleType(name)
        spec = _mach.ModuleSpec(name, loader=None, is_package=is_pkg)
        mod.__spec__ = spec
        mod.__loader__ = None
        if is_pkg:
            mod.__path__ = []
            mod.__package__ = name
        else:
            mod.__package__ = name.rpartition(".")[0]
        return mod

    apex_mod = _mk("apex", is_pkg=True)
    norm_mod = _mk("apex.normalization", is_pkg=True)
    fln_mod  = _mk("apex.normalization.fused_layer_norm", is_pkg=False)
    flnc_mod = _mk("fused_layer_norm_cuda", is_pkg=False)

    norm_mod.FusedRMSNorm   = _PyRMSNorm
    norm_mod.FusedLayerNorm = _PyFusedLayerNorm
    fln_mod.FusedRMSNorm    = _PyRMSNorm
    fln_mod.FusedLayerNorm  = _PyFusedLayerNorm
    apex_mod.normalization  = norm_mod

    _sys.modules["apex"] = apex_mod
    _sys.modules["apex.normalization"] = norm_mod
    _sys.modules["apex.normalization.fused_layer_norm"] = fln_mod
    _sys.modules["fused_layer_norm_cuda"] = flnc_mod


# =============================================================================
# Worker
#  - For 1-GPU replicas: gpu_ids = [g], we pin CUDA_VISIBLE_DEVICES and .cuda().
#  - For multi-GPU replicas (78B bf16): gpu_ids = [g0, g1, ...], we pin the
#    visible set to those physical IDs and let HF `device_map="auto"` shard
#    the model layers across the visible cards (pipeline parallel).
# =============================================================================
def _worker_main(
    worker_id: int,
    gpu_ids: List[int],
    model_path: str,
    quant: str,
    multi_gpu: bool,
    shard: Dict[str, Dict],
    n_gen: int,
    gen_kwargs: dict,
    progress_q: "mp.Queue",
    result_q: "mp.Queue",
):
    # Pin BEFORE importing torch CUDA APIs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    try:
        import torch as _torch
        # 1) Import transformers FIRST, so its apex probe sees real env (apex absent).
        from transformers import AutoModel, AutoTokenizer
        # 2) THEN install fake apex modules for InternVL's modeling code.
        _bypass_apex()

        dtype = _torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
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
        elif multi_gpu:
            # bf16 + multi-GPU per replica (78B): let HF shard layers.
            load_kwargs["device_map"] = "auto"
        # else: 1-GPU bf16 -> load to CPU then .cuda()

        model = AutoModel.from_pretrained(model_path, **load_kwargs).eval()
        if quant == "bf16" and not multi_gpu:
            model = model.cuda()

        # Which device should pixel_values go to?
        #  - 1-GPU: cuda:0 (only visible card)
        #  - multi-GPU bf16: the device of the input embedding layer (first stage)
        if multi_gpu:
            try:
                # InternVL: embeddings live under language_model
                input_device = next(model.language_model.get_input_embeddings().parameters()).device
            except Exception:
                input_device = _torch.device("cuda:0")
        else:
            input_device = _torch.device("cuda:0")

        def _normalize_paths(image_path) -> List[str]:
            if isinstance(image_path, str):
                return [image_path]
            if isinstance(image_path, (list, tuple)) and image_path:
                return list(image_path)
            raise ValueError(f"Invalid image_path: {image_path!r}")

        def _build_question(prompt: str, n_images: int) -> str:
            if n_images == 1:
                return f"<image>\n{prompt}"
            header = "".join(f"Image-{i+1}: <image>\n" for i in range(n_images))
            return f"{header}{prompt}"

        generation_config = dict(
            max_new_tokens=int(gen_kwargs.get("max_new_tokens", 1024)),
            do_sample=bool(gen_kwargs.get("do_sample", False)),
            temperature=float(gen_kwargs.get("temperature", 1.0)),
            top_p=float(gen_kwargs.get("top_p", 1.0)),
        )

        local_result: Dict[str, Dict] = {}
        for sid, sample in shard.items():
            try:
                paths = _normalize_paths(sample["image_path"])

                pv_list, num_patches_list = [], []
                for p in paths:
                    pv = _load_image_as_tiles(p, max_num=MAX_NUM_TILES)
                    pv_list.append(pv)
                    num_patches_list.append(pv.shape[0])

                pixel_values = _torch.cat(pv_list, dim=0).to(dtype).to(input_device)
                question = _build_question(sample["prompt"], len(paths))

                with _torch.inference_mode():
                    response = model.chat(
                        tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        question=question,
                        generation_config=generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=False,
                    )
                out_text = response.strip() if isinstance(response, str) else str(response)
            except Exception as e:
                out_text = f"[InternVL3-ERROR] {type(e).__name__}: {e}"

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
class InternVL3Inferencer:
    def __init__(
        self,
        model_size: str = "8b",
        quant: Optional[str] = None,
        gpus_per_replica: Optional[int] = None,
    ):
        key = model_size.lower()
        if key not in MODEL_PATHS:
            raise ValueError(f"Unknown model_size={model_size!r}. Supported: {SUPPORTED_SIZES}")
        self.model_size = key
        self.model_path = MODEL_PATHS[key]
        self.quant = (quant
                      or os.environ.get("INTERNVL3_QUANT")
                      or DEFAULT_QUANT[key]).lower()
        if self.quant not in ("bf16", "int8", "int4"):
            raise ValueError(f"quant must be bf16/int8/int4, got {self.quant!r}")

        env_gpr = os.environ.get("INTERNVL3_GPUS_PER_REPLICA")
        if gpus_per_replica is not None:
            self.gpus_per_replica = int(gpus_per_replica)
        elif env_gpr:
            self.gpus_per_replica = int(env_gpr)
        else:
            self.gpus_per_replica = DEFAULT_GPUS_PER_REPLICA[key]
        if self.gpus_per_replica < 1:
            raise ValueError(f"gpus_per_replica must be >= 1, got {self.gpus_per_replica}")
        # Quantized models load to a single device anyway -> force 1.
        if self.quant in ("int8", "int4") and self.gpus_per_replica != 1:
            print(f"[InternVL3][warn] quant={self.quant} forces gpus_per_replica=1 "
                  f"(was {self.gpus_per_replica}).")
            self.gpus_per_replica = 1

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

        # Partition physical GPUs into replicas of size `gpus_per_replica`.
        gpr = self.gpus_per_replica
        if gpr > len(phys_ids):
            raise RuntimeError(
                f"gpus_per_replica={gpr} > available GPUs ({len(phys_ids)}). "
                f"Either lower INTERNVL3_GPUS_PER_REPLICA or increase --gpu_n."
            )
        n_replicas = len(phys_ids) // gpr
        if n_replicas < 1:
            raise RuntimeError("Need at least 1 replica.")
        replica_gpus: List[List[int]] = [
            phys_ids[i * gpr:(i + 1) * gpr] for i in range(n_replicas)
        ]
        shards = self._shard(input, n_replicas)
        multi_gpu = (gpr > 1) and (self.quant == "bf16")

        ctx = mp.get_context("spawn")
        progress_q = ctx.Queue()
        result_q   = ctx.Queue()

        mode = f"PP{gpr}xDP{n_replicas}" if multi_gpu else f"DP{n_replicas} (1 GPU per replica)"
        print(f"[InternVL3] size={self.model_size} quant={self.quant} "
              f"{mode}  samples={len(input)}")
        for wid, (gids, sh) in enumerate(zip(replica_gpus, shards)):
            print(f"  replica {wid}: GPUs={gids}  shard_size={len(sh)}")

        procs = []
        for worker_id, (gpu_ids, shard) in enumerate(zip(replica_gpus, shards)):
            p = ctx.Process(
                target=_worker_main,
                args=(worker_id, gpu_ids, self.model_path, self.quant, multi_gpu,
                      shard, n_gen, gen_kwargs, progress_q, result_q),
                daemon=False,
            )
            p.start()
            procs.append(p)

        total = len(input)
        merged: Dict[str, Dict] = {}
        errors: List[str] = []
        finished_workers = 0
        desc = f"InternVL3-{self.model_size}[{self.quant}] {mode}"
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
            raise RuntimeError("InternVL3 worker(s) crashed:\n" + "\n".join(errors))
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
        m = InternVL3Inferencer(model_size="78b")  # bf16 by default, 4 GPUs / replica
        res = m.inference(small, gpu_n=8)
        print(json.dumps(res, ensure_ascii=False, indent=2))