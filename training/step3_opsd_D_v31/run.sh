#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

DATASET="$(prepare_dataset \
    step3_opsd_D_v31 \
    "$MVEI_DATA_ROOT/training_data/step3_opsd_D_v31/D7_C2_1_1.jsonl")"
START_MODEL="${MVEI_STEP2_MODEL:-$(latest_checkpoint "$MVEI_TRAIN_OUTPUT_ROOT/step2_grpo_C_v2")}" 
TEACHER_MODEL="${MVEI_STEP2_TEACHER_MODEL:-$START_MODEL}"
OUTPUT_DIR="$MVEI_TRAIN_OUTPUT_ROOT/step3_opsd_D_v31"
LOSS_PLUGIN="$SCRIPT_DIR/teacher_low_entropy.py"
export NCCL_DEBUG_FILE="$MVEI_TRAIN_OUTPUT_ROOT/logs/step3_opsd_D_v31_nccl.log"

export LOW_ENTROPY_TOP_RATIO="${LOW_ENTROPY_TOP_RATIO:-0.5}"
export LOW_ENTROPY_MIN_KEEP="${LOW_ENTROPY_MIN_KEEP:-1}"
export LOW_ENTROPY_WARMUP="${LOW_ENTROPY_WARMUP:-0}"
export DEBUG_LOW_ENTROPY="${DEBUG_LOW_ENTROPY:-0}"
export DEBUG_LOW_ENTROPY_EVERY="${DEBUG_LOW_ENTROPY_EVERY:-20}"
export DEBUG_LOW_ENTROPY_FILE="${DEBUG_LOW_ENTROPY_FILE:-$MVEI_TRAIN_OUTPUT_ROOT/logs/low_entropy_mask_debug.log}"
export DEEPSPEED_PASS_ENV="LOW_ENTROPY_TOP_RATIO,LOW_ENTROPY_MIN_KEEP,LOW_ENTROPY_WARMUP,DEBUG_LOW_ENTROPY,DEBUG_LOW_ENTROPY_EVERY,DEBUG_LOW_ENTROPY_FILE"

NPROC_PER_NODE="$GPUS_PER_NODE" "$SWIFT_BIN" rlhf \
    --rlhf_type gkd \
    --model "$START_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --dataset "$DATASET" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --lmbda 1.0 \
    --beta 0.5 \
    --seq_kd false \
    --temperature 1.0 \
    --top_p 1.0 \
    --gkd_logits_topk 128 \
    --sft_alpha 0 \
    --num_train_epochs 3 \
    --max_length 8192 \
    --max_completion_length 1024 \
    --max_pixels 262144 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-6 \
    --warmup_ratio 0.05 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": true}' \
    --freeze_vit true \
    --freeze_aligner false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.30 \
    --vllm_max_model_len 8192 \
    --vllm_enforce_eager true \
    --external_plugins "$LOSS_PLUGIN" \
    --log_completions true \
    --save_steps 5000 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --output_dir "$OUTPUT_DIR" \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 16 \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 4 \
    --dataloader_pin_memory true \
    --attn_impl flash_attn \
    --deepspeed "$DS_CONFIG"

