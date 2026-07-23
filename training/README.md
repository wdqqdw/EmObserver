# Four-stage EmObserver training pipeline

The training release is organized as four sequential stages:

```text
training/
├── config/ds_zero3_no_offload.json
├── step1_sft_C_v1/
├── step2_grpo_C_v2/
├── step3_opsd_D_v31/
└── step4_grpo_E_v1/
```

Each stage has its own `run.sh`. Reward or loss plugins used only by a particular stage are stored in that stage's directory.

## Data

Download the public dataset into `public_data/` as described in the repository README. The training subset is organized under:

```text
public_data/training_data/
├── step1_sft_C_v1/C_data.jsonl
├── step2_grpo_C_v2/A_1_10_data_grpo.jsonl
├── step3_opsd_D_v31/D7_C2_1_1.jsonl
├── step4_grpo_E_v1/E1_D7_p1k_grpo.jsonl
└── INSETS-462k/images/
```

The released JSONL files use portable `mvei-data://` image references. `prepare_data_paths.py` converts them to validated absolute paths immediately before each training job. Prepared files are written to `training_outputs/prepared_data/` and are not tracked by Git.

## Environment

The original experiments used eight A100 80 GB GPUs. Install the training dependencies in a dedicated environment:

```bash
python -m pip install -r training/requirements-training.txt
python -m pip install flash-attn --no-build-isolation
```

The ms-swift dependency is pinned to the original clean source commit. The supplied ZeRO-3 configuration is the only local ms-swift configuration required by these scripts.

Stage 1 starts from [Qwen/Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking). Download it into the default model directory with:

```bash
hf download Qwen/Qwen3-VL-8B-Thinking \
  --local-dir models/Qwen3-VL-8B-Thinking
```

Set the base Qwen3-VL model location if it is not stored at `models/Qwen3-VL-8B-Thinking`:

```bash
export MVEI_BASE_MODEL=/path/to/Qwen3-VL-8B-Thinking
```

Optional shared overrides include `MVEI_DATA_ROOT`, `MVEI_MODEL_ROOT`, `MVEI_TRAIN_OUTPUT_ROOT`, `GPUS_PER_NODE`, `CUDA_VISIBLE_DEVICES`, and `MASTER_PORT`.

## Run the stages

Run the stages from the repository root in order:

```bash
bash training/step1_sft_C_v1/run.sh
bash training/step2_grpo_C_v2/run.sh
bash training/step3_opsd_D_v31/run.sh
bash training/step4_grpo_E_v1/run.sh
```

Stages 2–4 automatically select the latest `checkpoint-*` directory from the preceding stage. To resume from or substitute an explicit checkpoint, set `MVEI_STEP1_MODEL`, `MVEI_STEP2_MODEL`, or `MVEI_STEP3_MODEL`. Step 3 uses the Step 2 checkpoint as both student initialization and fixed teacher unless `MVEI_STEP2_TEACHER_MODEL` is set.

## Step 4 judge API

Step 4 uses an Azure-compatible multimodal LLM judge. No API values are included in the repository. Supply your own configuration:

```bash
export MVEI_JUDGE_API_KEY="<your-api-key>"
export MVEI_JUDGE_API_VERSION="<your-api-version>"
export MVEI_JUDGE_ENDPOINT="<your-endpoint>"
export MVEI_JUDGE_MODEL="<your-deployment-or-model-name>"
```

`MVEI_JUDGE_LOG_ID` is optional. The Step 4 script exits before allocating GPUs when a required API variable is missing.
