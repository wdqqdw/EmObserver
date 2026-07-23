"""
InternVL3.5 inference wrapper (transformers + multi-GPU DATA parallelism, 1 GPU per replica).
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
    "8b":  model_path("InternVL3_5-8B"),
    "38b": model_path("InternVL3_5-38B"),
}
SUPPORTED_SIZES: List[str]= list(MODEL_PATHS.keys())

# Always 1 GPU per replica (pure DP).
TP_PER_REPLICA: Dict[str, int] = {"8b": 1, "38b": 1}

# Default precision per size. Override via env INTERNVL35_QUANT=bf16|int8|int4.
DEFAULT_QUANT: Dict[str, str] = {"8b": "bf16", "38b": "bf16"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_PATCH_SIZE = 448
MIN_NUM_TILES  = 1
MAX_NUM_TILES  = 12


# =============================================================================
# Image preprocessing (unchanged)
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

                pixel_values = _torch.cat(pv_list, dim=0).to(dtype).cuda()
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
                out_text = f"[InternVL35-ERROR] {type(e).__name__}: {e}"

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
# Public API (unchanged)
# =============================================================================
class InternVL35Inferencer:
    def __init__(self, model_size: str = "8b", quant: Optional[str] = None):
        key = model_size.lower()
        if key not in MODEL_PATHS:
            raise ValueError(f"Unknown model_size={model_size!r}. Supported: {SUPPORTED_SIZES}")
        self.model_size = key
        self.model_path = MODEL_PATHS[key]
        self.quant = (quant
                      or os.environ.get("INTERNVL35_QUANT")
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

        print(f"[InternVL3.5] size={self.model_size} quant={self.quant} "
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
        desc = f"InternVL3.5-{self.model_size}[{self.quant}] DP{n_replicas}"
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
            raise RuntimeError("InternVL3.5 worker(s) crashed:\n" + "\n".join(errors))
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
        m = InternVL35Inferencer(model_size="38b")  # bf16 by default
        res = m.inference(small, gpu_n=8)
        print(json.dumps(res, ensure_ascii=False, indent=2))