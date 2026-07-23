"""
DeepSeek-VL2 inference wrapper (transformers + multi-GPU data parallelism).

Repo  : https://github.com/deepseek-ai/DeepSeek-VL2  (pip install deepseek_vl2)
Models:
    small : ${MVEI_MODEL_ROOT}/deepseek-vl2-small   (16B MoE / A2.4B)
    base  : ${MVEI_MODEL_ROOT}/deepseek-vl2          (27B MoE / A4.5B)

Strategy
--------
* Each visible GPU loads its own full copy of the model. Input dict is
  round-robin sharded across workers; a shared tqdm bar aggregates progress.
* DeepSeek-VL2 chat format uses the explicit `<image>\\n` placeholder per
  image inside the user content; the processor rewrites these into image
  embedding slots.

Usage (identical interface to qwen3_vl_vllm.py):
    model = DeepSeekVL2Inferencer(model_size="small")
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
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from evaluate.infer_engines.paths import model_path


MODEL_PATHS: Dict[str, str] = {
    "small": model_path("deepseek-vl2-small"),
    "base":  model_path("deepseek-vl2"),
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
        from transformers import AutoModelForCausalLM
        # Official DeepSeek-VL2 helpers (pip install deepseek_vl2).
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        from deepseek_vl2.utils.io import load_pil_images

        device = _torch.device("cuda:0")  # only one visible -> always cuda:0
        dtype = _torch.bfloat16

        vl_processor = DeepseekVLV2Processor.from_pretrained(model_path)
        tokenizer = vl_processor.tokenizer

        # DeepSeek-VL2 ships its own ForCausalLM class; AutoModelForCausalLM
        # also works thanks to trust_remote_code, but the explicit class is
        # more robust against transformers auto-mapping changes.
        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        def _normalize_paths(image_path) -> List[str]:
            if isinstance(image_path, str):
                return [image_path]
            if isinstance(image_path, (list, tuple)) and image_path:
                return list(image_path)
            raise ValueError(f"Invalid image_path: {image_path!r}")

        def _build_conversation(prompt: str, image_paths: List[str]) -> List[dict]:
            """
            DeepSeek-VL2 chat format.
              user content = "<image>\\n" * N + prompt
              processor.format() will splice in the image embeddings.
            """
            placeholders = "".join(["<image>\n"] * len(image_paths))
            return [
                {
                    "role": "<|User|>",
                    "content": f"{placeholders}{prompt}",
                    "images": image_paths,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

        local_result: Dict[str, Dict] = {}
        for sid, sample in shard.items():
            try:
                paths = _normalize_paths(sample["image_path"])
                conversation = _build_conversation(sample["prompt"], paths)

                pil_images = load_pil_images(conversation)
                prepared = vl_processor(
                    conversations=conversation,
                    images=pil_images,
                    force_batchify=True,
                    system_prompt="",
                ).to(device)

                with _torch.inference_mode():
                    inputs_embeds = model.prepare_inputs_embeds(**prepared)
                    generated_ids = model.language.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepared.attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=int(gen_kwargs.get("max_new_tokens", 1024)),
                        do_sample=bool(gen_kwargs.get("do_sample", False)),
                        temperature=float(gen_kwargs.get("temperature", 1.0)),
                        top_p=float(gen_kwargs.get("top_p", 1.0)),
                        use_cache=True,
                    )

                # generate(inputs_embeds=...) returns ONLY new tokens.
                output_text = tokenizer.decode(
                    generated_ids[0].cpu().tolist(),
                    skip_special_tokens=True,
                ).strip()

            except Exception as e:
                output_text = f"[DeepSeekVL2-ERROR] {type(e).__name__}: {e}"

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
class DeepSeekVL2Inferencer:
    def __init__(self, model_size: str = "small"):
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
        with tqdm(total=total, desc=f"DeepSeekVL2-{self.model_size} x{gpu_n}",
                  dynamic_ncols=True) as pbar:
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
            raise RuntimeError("DeepSeekVL2 worker(s) crashed:\n" + "\n".join(errors))
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
        m = DeepSeekVL2Inferencer(model_size="small")
        res = m.inference(small, gpu_n=2)
        print(json.dumps(res, ensure_ascii=False, indent=2))