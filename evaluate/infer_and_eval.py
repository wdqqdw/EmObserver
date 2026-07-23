# protobuf                                 6.33.6
# evaluate/infer_and_eval.py
"""
Unified entry program for inference + metric evaluation across benchmarks.

Supported benchmarks (handled internally; default = ALL):
    - MVEI
    - EEmo-Bench-Single
    - EEmo-Bench-Pair
    - VECBench

------------------------------------------------------------------
Example commands (using default --bench and default --gpu_n):
------------------------------------------------------------------

# === BLIP2 === (Frequently meet AcceleratorError)
python -m evaluate.infer_and_eval --engine blip2           --size opt_6_7b

# pip install transformers==4.56.0 && pip install --user "protobuf==3.20.3" --force-reinstall --no-deps
# pip install --user --break-system-packages git+https://github.com/deepseek-ai/DeepSeek-VL.git
# === DeepSeek-VL === 
python -m evaluate.infer_and_eval --engine deepseek_vl     --size 7b_chat

# pip install transformers==4.45.2
# === mPLUG-Owl3 ===
python -m evaluate.infer_and_eval --engine mplug_owl3    --size 7b_240728

# pip install transformers==4.56.0 && pip install --user "protobuf==3.20.3" --force-reinstall --no-deps
# pip install --user --break-system-packages git+https://github.com/deepseek-ai/DeepSeek-VL2.git
# === DeepSeek-VL2 ===
python -m evaluate.infer_and_eval --engine deepseek_vl2    --size small
python -m evaluate.infer_and_eval --engine deepseek_vl2    --size base

# === MiMo-VL (vLLM) ===
python -m evaluate.infer_and_eval --engine mimo_vl_vllm      --size 7b

# pip install transformers==4.56.0 && pip install --user "protobuf==3.20.3" --force-reinstall --no-deps
# === InternVL3 ===
python -m evaluate.infer_and_eval --engine internvl3 --size 8b
python -m evaluate.infer_and_eval --engine internvl3 --size 38b  --gpu_n 8
python -m evaluate.infer_and_eval --engine internvl3 --size 78b  --gpu_n 8

# pip install transformers==4.56.0 && pip install --user "protobuf==3.20.3" --force-reinstall --no-deps
# === InternVL3.5 ===
python -m evaluate.infer_and_eval --engine internvl3_5 --size 8b
python -m evaluate.infer_and_eval --engine internvl3_5 --size 38b  --gpu_n 8

# === MiniCPM-V ===
python -m evaluate.infer_and_eval --engine minicpm_v --size 2_6
python -m evaluate.infer_and_eval --engine minicpm_v --size 4_5

# pip install transformers==4.56.0 && pip install --user "protobuf==3.20.3"
# === GLM-4.1 V ===
python -m evaluate.infer_and_eval --engine glm41v --size 4_1v_thinking

# pip install -U "transformers>=4.57.1"
# === GLM-4.6 V ===
python -m evaluate.infer_and_eval --engine glm46v --size 4_6v_flash    

# === Qwen3-VL (vLLM) ===
python -m evaluate.infer_and_eval --engine qwen3_vl_vllm   --size 8b_instruct
python -m evaluate.infer_and_eval --engine qwen3_vl_vllm   --size 8b_think
python -m evaluate.infer_and_eval --engine qwen3_vl_vllm   --size 32b_instruct
python -m evaluate.infer_and_eval --engine qwen3_vl_vllm   --size 32b_think
python -m evaluate.infer_and_eval --engine qwen3_vl_vllm   --size emobserver --gpu_n 8

# === Qwen2.5-VL (vLLM) ===
python -m evaluate.infer_and_eval --engine qwen25_vl_vllm  --size 7b_instruct   --gpu_n 4
python -m evaluate.infer_and_eval --engine qwen25_vl_vllm  --size 32b_instruct  --gpu_n 8
python -m evaluate.infer_and_eval --engine qwen25_vl_vllm  --size 72b_instruct  --gpu_n 8
python -m evaluate.infer_and_eval --engine qwen25_vl_vllm  --size emocaliber    --gpu_n 4
python -m evaluate.infer_and_eval --engine qwen25_vl_vllm  --size emovit        --gpu_n 4

# === Qwen3.5 (vLLM) ===
python -m evaluate.infer_and_eval --engine qwen35_vllm --size 9b   
python -m evaluate.infer_and_eval --engine qwen35_vllm --size 27b  

# === Qwen3.6 (vLLM) ===
python -m evaluate.infer_and_eval --engine qwen3_6_vllm --size 27b --gpu_n 8

# === Gemma-3 / Gemma-4 (transformers, multi-GPU DP) ===
python -m evaluate.infer_and_eval --engine gemma_mm --size 3_27b_it --gpu_n 8
python -m evaluate.infer_and_eval --engine gemma_mm --size 4_31b_it --gpu_n 8

# pip install -U "transformers>=4.57.1"
# === LLaVA-OneVision-2 ===
python -m evaluate.infer_and_eval --engine llava_ov2 --size 8b_instruct --gpu_n 8

# === Emotion-Qwen === (Do not supported by the default environment)
python -m evaluate.infer_and_eval --engine emotion_qwen  --size default    --gpu_n 8

# === GPT-5.5 (API) ===
python -m evaluate.infer_and_eval --engine gpt_5_5_api     --size 5_5

# === Qwen3.6-Plus (API) ===
python -m evaluate.infer_and_eval --engine qwen36_plus_api --size 3_6_plus
python -m evaluate.infer_and_eval --engine qwen36_plus_api --size kimi-k2.5

# === Seed-2.0-Pro (API) ===
python -m evaluate.infer_and_eval --engine seed_2_0_pro_api --size 2_0_pro
"""

import argparse
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

# ----------------------------------------------------------------------------- 
# Global registries
# ----------------------------------------------------------------------------- 

# Portable release layout:
#
#   public_code/
#   ├── evaluate/
#   ├── public_data/   # downloaded from wudq/MVEI_PLUS
#   └── outputs/       # generated inference results
#
# Both locations can be overridden without editing this file.
PUBLIC_CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("MVEI_DATA_ROOT", PUBLIC_CODE_ROOT / "public_data")
).expanduser().resolve()
BENCH_ROOT = str(DATA_ROOT / "benchmarks")
BASELINE_ROOT = str(
    Path(
        os.environ.get("MVEI_OUTPUT_ROOT", PUBLIC_CODE_ROOT / "outputs")
    ).expanduser().resolve()
)

BENCHMARKS: Dict[str, str] = {
    "MVEI":                            f"{BENCH_ROOT}/MVEI/Std-MVEI.json",
    "EEmo-Bench-Single-Perception":    f"{BENCH_ROOT}/EEmo-Bench/Std-EEmo-Bench-Single-Perception.json",
    "EEmo-Bench-Pair-Perception":      f"{BENCH_ROOT}/EEmo-Bench/Std-EEmo-Bench-Pair-Perception.json",
    "VECBench":                        f"{BENCH_ROOT}/VECBench/Std-VECBench.json",
}

METRIC_CALCULATORS: Dict[str, dict] = {
    "MVEI": {
        "module": "evaluate.metrics_calculators.mvei",
        "class":  "MVEIEvaluator",
    },
    "VECBench": {
        "module": "evaluate.metrics_calculators.vecbench",
        "class":  "VECBenchEvaluator",
    },
    "EEmo-Bench-Single-Perception": {
        "module": "evaluate.metrics_calculators.eemo_bench",
        "class":  "EEmoBenchEvaluator",
        "kwargs": {"subset": "single"},
        "merge_into": "EEmo-Bench",
    },
    "EEmo-Bench-Pair-Perception": {
        "module": "evaluate.metrics_calculators.eemo_bench",
        "class":  "EEmoBenchEvaluator",
        "kwargs": {"subset": "pair"},
        "merge_into": "EEmo-Bench",
    },
}

# engine_name -> spec.
#   "kind" : 'vllm' | 'api'  -- governs how SamplingParams is built.
INFER_ENGINES: Dict[str, dict] = {
    "blip2": {
        "module":         "evaluate.infer_engines.blip2",
        "class":          "BLIP2Inferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/BLIP2-{{size_pretty}}",
        "kind":           "api",   # 用 SimpleNamespace 喂 sampling_params
    },
    "deepseek_vl": {
        "module":         "evaluate.infer_engines.deepseek_vl",
        "class":          "DeepSeekVLInferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/DeepSeek-VL-{{size_pretty}}",
        "kind":           "api",
    },
    "mplug_owl3": {
        "module":         "evaluate.infer_engines.mplug_owl3",
        "class":          "MPlugOwl3Inferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/mPLUG-Owl3-{{size_pretty}}",
        "kind":           "api",
    },
    "deepseek_vl2": {
        "module":         "evaluate.infer_engines.deepseek_vl2",
        "class":          "DeepSeekVL2Inferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/DeepSeek-VL2-{{size_pretty}}",
        "kind":           "api",
    },
    "mimo_vl_vllm": {
        "module":         "evaluate.infer_engines.mimo_vl_vllm",
        "class":          "MiMoVLInferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/MiMo-VL-{{size_pretty}}",
        "kind":           "vllm",
    },
    "internvl3": {
        "module":         "evaluate.infer_engines.internvl3",
        "class":          "InternVL3Inferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/InternVL3-{{size_pretty}}",
        "kind":           "api",
    },
    "internvl3_5": {
        "module":         "evaluate.infer_engines.internvl3_5",
        "class":          "InternVL35Inferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/InternVL3.5-{{size_pretty}}",
        "kind":           "api",
    },
    "minicpm_v": {
        "module":         "evaluate.infer_engines.minicpm_v",
        "class":          "MiniCPMVInferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/MiniCPM-V-{{size_pretty}}",
        "kind":           "api",
    },
    "glm41v": {
        "module":         "evaluate.infer_engines.glm41v", 
        "class":          "GLM41VInferencer",
        "kind":           "api",
        "output_dir_tpl": f"{BASELINE_ROOT}/GLM-4_1v-{{size_pretty}}",
    },
    "glm46v": {
        "module":         "evaluate.infer_engines.glm46v", 
        "class":          "GLM46VInferencer",
        "kind":           "api",
        "output_dir_tpl": f"{BASELINE_ROOT}/GLM-4_6v-{{size_pretty}}",
    },
    "qwen3_vl_vllm": {
        "module":         "evaluate.infer_engines.qwen3_vl_vllm",
        "class":          "Qwen3VLInferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/Qwen3-VL-{{size_pretty}}",
        "kind":           "vllm",
    },
    "qwen25_vl_vllm": {                                                      
        "module":         "evaluate.infer_engines.qwen25_vl_vllm", 
        "class":          "Qwen25VLInferencer",                             
        "output_dir_tpl": f"{BASELINE_ROOT}/Qwen2.5-VL-{{size_pretty}}",      
        "kind":           "vllm",                                             
    },              
    "qwen35_vllm": {
        "module":         "evaluate.infer_engines.qwen35_vllm",
        "class":          "Qwen35Inferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/Qwen3.5-{{size_pretty}}",
        "kind":           "vllm",
    },                
    "qwen3_6_vllm": {
        "module":         "evaluate.infer_engines.qwen36_vllm",
        "class":          "Qwen36Inferencer",
        "kind":           "vllm",
        "output_dir_tpl": f"{BASELINE_ROOT}/Qwen3.6-{{size_pretty}}",
    },          
    "llava_ov2": {
        "module":         "evaluate.infer_engines.llava_onevision2",
        "class":          "LLaVAOneVision2Inferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/LLaVA-OneVision-2-{{size_pretty}}",
        "kind":           "api",   # 走 SimpleNamespace sampling_params
    },      
    "gemma_mm": {
        "module":         "evaluate.infer_engines.gemma3",
        "class":          "Gemma3Inferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/Gemma-{{size_pretty}}",
        "kind":           "api",   # 走 SimpleNamespace sampling_params（与 InternVL 一致）
    },                          
    "gpt_5_5_api": {
        "module":         "evaluate.infer_engines.gpt_5_5_api",
        "class":          "GPT55Inferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/GPT-{{size_pretty}}",
        "kind":           "api",
    },
    "qwen36_plus_api": {
        "module":         "evaluate.infer_engines.qwen36_plus_api",
        "class":          "Qwen36PlusInferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/Qwen3.6-{{size_pretty}}",
        "kind":           "api",
    },
    "seed_2_0_pro_api": {
        "module":         "evaluate.infer_engines.seed_20_pro_api",
        "class":          "Seed20ProInferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/Seed-{{size_pretty}}",
        "kind":           "api",
    },
    "emotion_qwen": {
        "module":         "evaluate.infer_engines.emotion_qwen",
        "class":          "EmotionQwenInferencer",
        "output_dir_tpl": f"{BASELINE_ROOT}/EmotionQwen-{{size_pretty}}",
        "kind":           "api",
    },
}


# ----------------------------------------------------------------------------- 
# Helpers
# ----------------------------------------------------------------------------- 

def _pretty_size(size: str) -> str:
    """
    Normalize a size tag for use in directory names.
        '8b_instruct' -> '8B_Instruct'
        '5_5'         -> '5_5'
        '3_6_plus'    -> '3_6_Plus'
    """
    parts = size.split("_")
    out = []
    for i, p in enumerate(parts):
        if i == 0:
            out.append(p.upper())
        else:
            out.append(p.capitalize())
    return "_".join(out)


def resolve_output_dir(engine_name: str, size: str) -> str:
    spec = INFER_ENGINES[engine_name]
    return spec["output_dir_tpl"].format(size=size, size_pretty=_pretty_size(size))


def validate_data_layout(bench_ids):
    """Fail early with an actionable message when public data is not assembled."""
    unknown = [bid for bid in bench_ids if bid not in BENCHMARKS]
    if unknown:
        raise KeyError(f"Unknown benchmark ids: {unknown}. Known: {list(BENCHMARKS)}")
    missing = [BENCHMARKS[bid] for bid in bench_ids if not Path(BENCHMARKS[bid]).is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required benchmark metadata was not found:\n"
            f"{formatted}\n\n"
            "Download the public data from Hugging Face at the repository root:\n"
            "  hf download wudq/MVEI_PLUS --repo-type dataset --local-dir public_data\n"
            "or set MVEI_DATA_ROOT to the downloaded data directory."
        )


def _resolve_image_paths(record):
    """Resolve release metadata paths against the downloaded benchmark root."""
    record = dict(record)
    values = record.get("image_path")
    if values is None:
        return record
    is_list = isinstance(values, list)
    paths = values if is_list else [values]
    resolved = []
    for value in paths:
        path_text = str(value).replace("\\", "/")
        path = Path(path_text)
        if path.is_absolute():
            resolved.append(str(path))
            continue
        prefix = "../benchmarks/"
        if path_text.startswith(prefix):
            path = Path(BENCH_ROOT) / path_text[len(prefix):]
        else:
            path = DATA_ROOT / path_text
        resolved.append(str(path.resolve()))
    record["image_path"] = resolved if is_list else resolved[0]
    return record


def load_and_merge(bench_ids):
    merged = {}
    per_bench_keys = {}
    for bid in bench_ids:
        if bid not in BENCHMARKS:
            raise KeyError(f"Unknown benchmark id: {bid!r}. Known: {list(BENCHMARKS)}")
        with open(BENCHMARKS[bid], "r", encoding="utf-8") as f:
            data = json.load(f)
        keys = []
        for sid, rec in data.items():
            mkey = f"{bid}::{sid}"
            if mkey in merged:
                raise ValueError(f"Duplicate merged key: {mkey}")
            merged[mkey] = _resolve_image_paths(rec)
            keys.append(mkey)
        per_bench_keys[bid] = keys
        print(f"[load] {bid}: {len(keys)} samples")
    print(f"[load] total merged: {len(merged)}")
    return merged, per_bench_keys


def split_results(results, per_bench_keys):
    out = {bid: {} for bid in per_bench_keys}
    for bid, keys in per_bench_keys.items():
        prefix = f"{bid}::"
        for mkey in keys:
            if mkey not in results:
                print(f"[warn] missing inference result for {mkey}")
                continue
            orig_id = mkey[len(prefix):]
            out[bid][orig_id] = results[mkey]
    return out


def build_engine(engine_name: str, model_size: str):
    if engine_name not in INFER_ENGINES:
        raise KeyError(f"Unknown engine: {engine_name!r}. Known: {list(INFER_ENGINES)}")
    spec = INFER_ENGINES[engine_name]
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["class"])
    output_dir = resolve_output_dir(engine_name, model_size)
    return cls(model_size=model_size), output_dir, spec.get("kind", "vllm")


def build_evaluator_spec(bench_id: str):
    spec = METRIC_CALCULATORS.get(bench_id)
    if spec is None:
        return None, None, None
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["class"])
    return cls(), spec.get("kwargs", {}), spec.get("merge_into", bench_id)


def _supported_sizes_for(engine_name: str):
    try:
        mod = importlib.import_module(INFER_ENGINES[engine_name]["module"])
        return getattr(mod, "SUPPORTED_SIZES", [])
    except Exception:
        return []


def _make_sampling_params(kind: str, n: int, temp: float, max_tokens: int):
    """vLLM kind needs the real SamplingParams; API kind uses a duck-typed namespace."""
    if kind == "vllm":
        from vllm import SamplingParams
        return SamplingParams(n=n, temperature=temp, max_tokens=max_tokens)
    return SimpleNamespace(n=n, temperature=temp, max_tokens=max_tokens)


def _merge_metrics(dst: Dict, src: Dict):
    for k, v in src.items():
        if k not in dst or dst[k] is None:
            dst[k] = v
        elif v is not None:
            print(f"[warn] metric key {k!r} present in multiple sources; overwriting")
            dst[k] = v


# ----------------------------------------------------------------------------- 
# Main
# ----------------------------------------------------------------------------- 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench",  nargs="+",
                        default=["MVEI", "VECBench",
                                 "EEmo-Bench-Single-Perception",
                                 "EEmo-Bench-Pair-Perception"],
                        help=f"benchmark ids to run, available: {list(BENCHMARKS)}")
    default_engine = "gpt_5_5_api"
    parser.add_argument("--engine", default=default_engine,
                        help=f"inference engine, one of: {list(INFER_ENGINES)}")
    default_sizes = _supported_sizes_for(default_engine)
    parser.add_argument(
        "--size", default="5_5",
        help=f"model size tag (engine-specific). {default_engine} supports: {default_sizes}"
    )
    parser.add_argument("--gpu_n",  type=int, default=None,
                        help="num GPUs (default: all visible). Ignored by API engines.")
    parser.add_argument("--n",      type=int, default=1, help="rollouts per sample")
    parser.add_argument("--temp",   type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    # 1) Load + merge
    validate_data_layout(args.bench)
    merged, per_bench_keys = load_and_merge(args.bench)

    # 2) Build engine and sampling params
    engine, output_dir, kind = build_engine(args.engine, args.size)
    print(f"[engine] {args.engine}/{args.size} (kind={kind}) -> output_dir={output_dir}")
    sp = _make_sampling_params(kind, args.n, args.temp, args.max_tokens)

    # 3) Inference on the merged dict
    results = engine.inference(merged, gpu_n=args.gpu_n, sampling_params=sp)

    # 4) Split and save per benchmark
    os.makedirs(output_dir, exist_ok=True)
    per_bench_results = split_results(results, per_bench_keys)
    saved_paths: Dict[str, str] = {}
    for bid, recs in per_bench_results.items():
        out_name = f"Std_{bid.replace('-', '_')}_results.json"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        saved_paths[bid] = out_path
        print(f"[save] {bid}: {len(recs)} -> {out_path}")

    # 5) Run metric calculators on the saved files
    all_metrics: Dict[str, Dict] = {}
    for bid, path in saved_paths.items():
        evaluator, kwargs, merge_key = build_evaluator_spec(bid)
        if evaluator is None:
            print(f"[eval] no calculator registered for {bid}, skip")
            continue
        metrics = evaluator.evaluate(path, **kwargs)
        if merge_key in all_metrics:
            _merge_metrics(all_metrics[merge_key], metrics)
        else:
            all_metrics[merge_key] = dict(metrics)
        print(f"[eval] {bid}: {metrics}")

    if all_metrics:
        summary_path = os.path.join(output_dir, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"[summary] -> {summary_path}")


if __name__ == "__main__":
    main()
