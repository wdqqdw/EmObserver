#!/usr/bin/env bash

set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TRAINING_ROOT/.." && pwd)"

export MVEI_DATA_ROOT="${MVEI_DATA_ROOT:-$REPO_ROOT/public_data}"
export MVEI_MODEL_ROOT="${MVEI_MODEL_ROOT:-$REPO_ROOT/models}"
export MVEI_TRAIN_OUTPUT_ROOT="${MVEI_TRAIN_OUTPUT_ROOT:-$REPO_ROOT/training_outputs}"

export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
if [[ -z "${MASTER_PORT:-}" ]]; then
    if [[ -n "${METIS_WORKER_0_PORT:-}" ]]; then
        export MASTER_PORT="${METIS_WORKER_0_PORT%%,*}"
    else
        export MASTER_PORT=29500
    fi
fi
export RANK="${RANK:-0}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCH_CHECKPOINT_SERIALIZATION="${TORCH_CHECKPOINT_SERIALIZATION:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

SWIFT_BIN="${SWIFT_BIN:-swift}"
DS_CONFIG="$TRAINING_ROOT/config/ds_zero3_no_offload.json"

mkdir -p \
    "$MVEI_TRAIN_OUTPUT_ROOT/logs" \
    "$MVEI_TRAIN_OUTPUT_ROOT/prepared_data"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

require_env_vars() {
    local missing=()
    local name
    for name in "$@"; do
        if [[ -z "${!name:-}" ]]; then
            missing+=("$name")
        fi
    done
    if (( ${#missing[@]} > 0 )); then
        echo "Missing required environment variables: ${missing[*]}" >&2
        exit 1
    fi
}

prepare_dataset() {
    local step_name="$1"
    local source_path="$2"
    local output_path="$MVEI_TRAIN_OUTPUT_ROOT/prepared_data/${step_name}.jsonl"
    require_file "$source_path"
    python3 "$TRAINING_ROOT/prepare_data_paths.py" \
        --input "$source_path" \
        --output "$output_path" \
        --data-root "$MVEI_DATA_ROOT" >&2
    echo "$output_path"
}

latest_checkpoint() {
    local stage_root="$1"
    local checkpoint
    checkpoint="$(find "$stage_root" -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1)"
    if [[ -z "$checkpoint" ]]; then
        echo "No checkpoint found below $stage_root" >&2
        echo "Run the previous stage first or set its MVEI_STEP*_MODEL override." >&2
        exit 1
    fi
    echo "$checkpoint"
}

require_file "$DS_CONFIG"
cd "$REPO_ROOT"

