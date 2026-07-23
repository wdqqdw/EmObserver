#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

DATASET="$(prepare_dataset \
    step1_sft_C_v1 \
    "$MVEI_DATA_ROOT/training_data/step1_sft_C_v1/C_data.jsonl")"
MODEL="${MVEI_BASE_MODEL:-$MVEI_MODEL_ROOT/Qwen3-VL-8B-Thinking}"
OUTPUT_DIR="$MVEI_TRAIN_OUTPUT_ROOT/step1_sft_C_v1"
export NCCL_DEBUG_FILE="$MVEI_TRAIN_OUTPUT_ROOT/logs/step1_sft_C_v1_nccl.log"

NPROC_PER_NODE="$GPUS_PER_NODE" "$SWIFT_BIN" sft \
    --model "$MODEL" \
    --tuner_type full \
    --dataset "$DATASET" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --gradient_accumulation_steps 1 \
    --eval_steps 5000 \
    --save_steps 5000 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 8192 \
    --max_pixels 802816 \
    --freeze_vit true \
    --freeze_aligner false \
    --output_dir "$OUTPUT_DIR" \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataloader_persistent_workers true \
    --dataloader_pin_memory true \
    --dataloader_prefetch_factor 4 \
    --attn_impl flash_attn \
    --deepspeed "$DS_CONFIG"

