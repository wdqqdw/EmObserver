#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

DATASET="$(prepare_dataset \
    step2_grpo_C_v2 \
    "$MVEI_DATA_ROOT/training_data/step2_grpo_C_v2/A_1_10_data_grpo.jsonl")"
START_MODEL="${MVEI_STEP1_MODEL:-$(latest_checkpoint "$MVEI_TRAIN_OUTPUT_ROOT/step1_sft_C_v1")}" 
OUTPUT_DIR="$MVEI_TRAIN_OUTPUT_ROOT/step2_grpo_C_v2"
REWARD_PLUGIN="$SCRIPT_DIR/rewards.py"
export NCCL_DEBUG_FILE="$MVEI_TRAIN_OUTPUT_ROOT/logs/step2_grpo_C_v2_nccl.log"

NPROC_PER_NODE="$GPUS_PER_NODE" "$SWIFT_BIN" rlhf \
    --rlhf_type grpo \
    --model "$START_MODEL" \
    --dataset "$DATASET" \
    --tuner_type full \
    --external_plugins "$REWARD_PLUGIN" \
    --reward_funcs mvei_accuracy mvei_format \
    --reward_weights 1.0 0.5 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_length 8192 \
    --max_completion_length 1024 \
    --max_pixels 262144 \
    --num_generations 4 \
    --steps_per_generation 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --temperature 1.0 \
    --top_p 1.0 \
    --epsilon 0.2 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": true}' \
    --freeze_vit true \
    --freeze_aligner false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.45 \
    --vllm_max_model_len 8192 \
    --vllm_enforce_eager true \
    --log_completions true \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --output_dir "$OUTPUT_DIR" \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 4 \
    --dataloader_pin_memory true \
    --attn_impl flash_attn \
    --deepspeed "$DS_CONFIG"

