#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

require_env_vars \
    MVEI_JUDGE_API_KEY \
    MVEI_JUDGE_API_VERSION \
    MVEI_JUDGE_ENDPOINT \
    MVEI_JUDGE_MODEL

DATASET="$(prepare_dataset \
    step4_grpo_E_v1 \
    "$MVEI_DATA_ROOT/training_data/step4_grpo_E_v1/E1_D7_p1k_grpo.jsonl")"
START_MODEL="${MVEI_STEP3_MODEL:-$(latest_checkpoint "$MVEI_TRAIN_OUTPUT_ROOT/step3_opsd_D_v31")}" 
OUTPUT_DIR="$MVEI_TRAIN_OUTPUT_ROOT/step4_grpo_E_v1"
export NCCL_DEBUG_FILE="$MVEI_TRAIN_OUTPUT_ROOT/logs/step4_grpo_E_v1_nccl.log"

NPROC_PER_NODE="$GPUS_PER_NODE" "$SWIFT_BIN" rlhf \
    --rlhf_type grpo \
    --model "$START_MODEL" \
    --dataset "$DATASET" \
    --tuner_type full \
    --external_plugins \
        "$SCRIPT_DIR/rewards.py" \
        "$SCRIPT_DIR/llm_consistency.py" \
    --reward_funcs mvei_accuracy mvei_format mvei_llm_consistency \
    --reward_weights 1.0 0.5 1.0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_length 8192 \
    --max_completion_length 1024 \
    --max_pixels 262144 \
    --num_generations 8 \
    --steps_per_generation 1 \
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
    --save_steps 200 \
    --save_total_limit 10 \
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

