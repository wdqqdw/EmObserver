# EmObserver: Official Implementation

Official code repository of "MVEI & EmObserver: Empowering MLLM-Oriented Visual Emotional Intelligence via Emotion Statement Judgement".

Inference, evaluation, and four-stage EmObserver training code for the MVEI Expansion project. Data artifacts are released separately through [wudq/MVEI_PLUS](https://huggingface.co/datasets/wudq/MVEI_PLUS), and the trained checkpoint is available at [wudq/EmObserver](https://huggingface.co/wudq/EmObserver).

## 🔗 Project map

| Resource | Relationship to this repository | Link |
| --- | --- | --- |
| Original conference version | Original MVEI code and project release | [wdqqdw/MVEI](https://github.com/wdqqdw/MVEI) |
| Original MVEI dataset | Source benchmark release | [wudq/MVEI](https://huggingface.co/datasets/wudq/MVEI) |
| Original INSETS-462k dataset | Source training-data release | [wudq/INSETS-462k](https://huggingface.co/datasets/wudq/INSETS-462k) |
| MVEI_PLUS | Expanded benchmarks, released predictions, four-stage training data, and the GPT-5.5-filtered INSETS-462k derivative | [wudq/MVEI_PLUS](https://huggingface.co/datasets/wudq/MVEI_PLUS) |
| EmObserver | Model checkpoint produced by the expanded training pipeline | [wudq/EmObserver](https://huggingface.co/wudq/EmObserver) |

This repository is the executable code companion to MVEI_PLUS. It extends the [original conference repository](https://github.com/wdqqdw/MVEI); it does not replace the original release. MVEI_PLUS also releases the GPT-5.5-filtered INSETS-462k version used here, while the complete source dataset remains at [wudq/INSETS-462k](https://huggingface.co/datasets/wudq/INSETS-462k).

## 🚀 Quick start

Run the following commands from the repository root.

### 1. Install the common dependencies

```bash
python3 -m pip install -r requirements.txt
```

GPU inference and training require a compatible PyTorch/CUDA stack; see the Environment and dependencies section below before installing vLLM or FlashAttention.

### 2. Download MVEI_PLUS

```bash
python3 -m pip install -U huggingface_hub
hf download wudq/MVEI_PLUS --repo-type dataset --local-dir public_data
```

The resulting repository layout should include:

```text
public_code/
├── evaluate/
├── training/
└── public_data/
    ├── baselines/
    ├── benchmarks/
    └── training_data/
```

The Hugging Face CLI may create `public_data/.cache/huggingface/`; this cache is ignored by Git and is not dataset content.

### 3. Download EmObserver

```bash
hf download wudq/EmObserver --local-dir models/EmObserver
```

### 4. Run EmObserver inference and evaluation

```bash
python3 -m evaluate.infer_and_eval \
  --engine qwen3_vl_vllm \
  --size emobserver \
  --gpu_n 8
```

Adjust `--gpu_n` to the visible GPUs that can hold the model. Results are written under `outputs/` by default.

## 📁 Repository layout

```text
public_code/
├── .gitignore
├── README.md
├── environment.txt      # sanitized experimental environment snapshot
├── requirements.txt     # minimal common dependencies
├── evaluate/
│   ├── infer_and_eval.py
│   ├── infer_engines/
│   └── metrics_calculators/
├── training/
│   ├── README.md
│   ├── step1_sft_C_v1/
│   ├── step2_grpo_C_v2/
│   ├── step3_opsd_D_v31/
│   └── step4_grpo_E_v1/
├── public_data/          # downloaded from Hugging Face; not stored in Git
├── models/               # local model weights; not stored in Git
├── outputs/              # evaluation outputs
└── training_outputs/     # prepared data and training checkpoints
```

## 🧪 Inference and evaluation

`evaluate/infer_and_eval.py` is the main entry point. It loads the requested standardized benchmark metadata, dynamically registers the selected inference engine, runs generation, separates the results by benchmark, and invokes the corresponding metric calculator.

Supported benchmark identifiers are:

- `MVEI`
- `VECBench`
- `EEmo-Bench-Single-Perception`
- `EEmo-Bench-Pair-Perception`

If `--bench` is omitted, all four are evaluated. To select a subset:

```bash
python3 -m evaluate.infer_and_eval \
  --engine <engine> \
  --size <size> \
  --gpu_n <number_of_gpus> \
  --bench MVEI VECBench
```

Run `python3 -m evaluate.infer_and_eval --help` for the registered engines and arguments. Model-specific size tags and optional dependencies are documented in the corresponding module under `evaluate/infer_engines/`. EmObserver uses engine `qwen3_vl_vllm` and size tag `emobserver`.

The `evaluate/metrics_calculators/` package contains the MVEI, EEmo-Bench, and VECBench metric implementations. They consume the per-benchmark result JSON files produced by the main entry point and write a combined `metrics_summary.json`.

## 🏋️ Four-stage training

The training pipeline is organized as four independently runnable stages:

| Stage | Directory | Method | Default input |
| ---: | --- | --- | --- |
| 1 | `training/step1_sft_C_v1/` | SFT | `public_data/training_data/step1_sft_C_v1/` |
| 2 | `training/step2_grpo_C_v2/` | GRPO | `public_data/training_data/step2_grpo_C_v2/` |
| 3 | `training/step3_opsd_D_v31/` | OPSD | `public_data/training_data/step3_opsd_D_v31/` |
| 4 | `training/step4_grpo_E_v1/` | GRPO with LLM-consistency reward | `public_data/training_data/step4_grpo_E_v1/` |

Download the default base model if it is not already available:

```bash
hf download Qwen/Qwen3-VL-8B-Thinking --local-dir models/Qwen3-VL-8B-Thinking
```

Then run each stage from the repository root:

```bash
bash training/step1_sft_C_v1/run.sh
bash training/step2_grpo_C_v2/run.sh
bash training/step3_opsd_D_v31/run.sh
bash training/step4_grpo_E_v1/run.sh
```

Each launcher validates its source JSONL and `mvei-data://` image references, writes a materialized stage-local JSONL under `training_outputs/prepared_data/`, and then starts training. Outputs are chained between stages through the defaults in the scripts; override model/checkpoint locations through the environment variables documented in [`training/README.md`](training/README.md).

Stage 4 uses an external LLM-consistency judge. Provide your own endpoint credentials before running it; no API key is included in this release. The detailed variable names and checkpoint-chain controls are listed in [`training/README.md`](training/README.md).

The separately released `public_data/training_data/INSETS-462k-GPT-5.5-filtered/A_data.jsonl` contains 301,911 GPT-5.5-filtered SFT records. It uses the same portable image convention and shared 14,760-image subset as the relevant training stages. See the [MVEI_PLUS dataset card](https://huggingface.co/datasets/wudq/MVEI_PLUS) for data schemas and examples.

## ⚙️ Environment and dependencies

`requirements.txt` contains the minimal common dependencies. `environment.txt` is a sanitized snapshot of the full experimental environment; ByteDance-internal repositories, packages, and infrastructure references have been removed. Because the snapshot includes transitive and system-specific packages, installing it wholesale into an existing environment is not recommended.

Key GPU-stack versions recorded by the snapshot are:

| Package | Snapshot version | Upstream repository |
| --- | ---: | --- |
| `transformers` | 5.9.0 | [huggingface/transformers](https://github.com/huggingface/transformers) |
| `vllm` | 0.17.1 | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| `flash-attn` | 2.8.3 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| `torch` | 2.10.0 | [pytorch/pytorch](https://github.com/pytorch/pytorch) |

PyTorch, CUDA, compiler, Transformers, vLLM, FlashAttention, and the target GPU architecture must be mutually compatible. Install the appropriate PyTorch/CUDA build first, then FlashAttention, vLLM, and the remaining packages. Revalidate inference and training after changing any one of these components.

Some inference engines need upstream model packages that are intentionally optional:

```bash
# DeepSeek-VL
python3 -m pip install "git+https://github.com/deepseek-ai/DeepSeek-VL.git"

# DeepSeek-VL2
python3 -m pip install "git+https://github.com/deepseek-ai/DeepSeek-VL2.git"
```

Install only the optional engine packages you plan to use when their version constraints conflict.

## 🧭 Data, model, and output paths

Defaults are repository-relative and contain no original-machine paths:

| Purpose | Default | Override |
| --- | --- | --- |
| Data root | `public_data/` | `MVEI_DATA_ROOT` |
| Model root | `models/` | `MVEI_MODEL_ROOT` |
| Evaluation output | `outputs/` | `MVEI_OUTPUT_ROOT` |
| Training output | `training_outputs/` | stage variables in `training/README.md` |

Example with external data and model storage:

```bash
export MVEI_DATA_ROOT=/path/to/MVEI_PLUS
export MVEI_MODEL_ROOT=/path/to/models
python3 -m evaluate.infer_and_eval --engine qwen3_vl_vllm --size emobserver --gpu_n 8
```

Individual model locations can be overridden with `MVEI_MODEL_<NORMALIZED_MODEL_NAME>`, for example:

```bash
export MVEI_MODEL_EMOBSERVER=/path/to/EmObserver
export MVEI_MODEL_QWEN3_VL_8B_INSTRUCT=/path/to/Qwen3-VL-8B-Instruct
```

## 🔑 API engines

Credentials are read only from environment variables. Every value is empty by default; configure your own API before selecting an API-backed engine.

| Engine | Required environment variables | Optional variables |
| --- | --- | --- |
| GPT-5.5 Azure wrapper | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_MODEL` | `AZURE_OPENAI_LOG_ID` |
| Qwen3.6-Plus/Kimi Azure-compatible wrapper | `QWEN_API_KEY`, `QWEN_API_VERSION`, `QWEN_AZURE_ENDPOINT`, and the selected model variable (`QWEN36_PLUS_MODEL` or `KIMI_K25_MODEL`) | `QWEN_LOG_ID` |
| Seed-2.0-Pro wrapper | `SEED_API_KEY`, `SEED_BASE_URL`, `SEED_MODEL` | none |

Example:

```bash
export AZURE_OPENAI_API_KEY="<your-api-key>"
export AZURE_OPENAI_API_VERSION="<your-api-version>"
export AZURE_OPENAI_ENDPOINT="<your-endpoint>"
export AZURE_OPENAI_MODEL="<your-deployment-or-model-name>"
```

If a required value is missing, the corresponding engine exits before inference and reports the missing variable names. The internal `r1_omni` engine is intentionally excluded from this release.

## ✅ Reproducibility checklist

Before evaluation:

- download [MVEI_PLUS](https://huggingface.co/datasets/wudq/MVEI_PLUS) so `public_data/benchmarks/` exists;
- download [EmObserver](https://huggingface.co/wudq/EmObserver), or configure another supported engine and checkpoint;
- install that engine's compatible GPU stack and optional dependencies;
- confirm the selected GPU count and available memory; and
- provide environment variables if using an API-backed engine.

Before training:

- confirm all four directories under `public_data/training_data/` are present;
- confirm the shared images are present under `public_data/training_data/INSETS-462k/images/` and the VECBench images are present for stage 1;
- download the base model or set the documented model override;
- run stages in order when reproducing the full checkpoint chain; and
- provide your own Stage-4 judge API settings.

Static packaging checks cover Python and shell syntax, benchmark and training-data path resolution, plugin registration against the pinned ms-swift commit, and no-model command construction. Full GPU inference and training are not executed as part of packaging because they require model weights, compatible accelerators, and—where applicable—user-owned API credentials.
