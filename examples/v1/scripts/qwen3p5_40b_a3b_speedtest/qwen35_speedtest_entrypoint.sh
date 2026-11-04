#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$SOURCE_ROOT"

# RJob supplies NODE_COUNT, NODE_RANK, MASTER_ADDR and PROC_PER_NODE.
export WORLD_SIZE=$((NODE_COUNT * PROC_PER_NODE))
export WORK_DIR="$RUN_DIR/train"
export PYTHONPATH="$SOURCE_ROOT"
export DATA_PATH="${DATA_PATH:-/mnt/shared-storage-user/linzhouhan/public_data/olmo-500B-qwen35-tokenized/data_args_path.txt}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/dolma3_mix-150B-1025/megatron_qwen35_8192/tokenizer/Qwen3.5-0.8B-Base}"
export XTUNER_USE_FA3=1
export XTUNER_MUON_USE_TRITON=1  # TODO (tzhu): check if this could be enabled to speedup
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton/$NODE_RANK"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache/inductor/$NODE_RANK"
export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda/$NODE_RANK"

mkdir -p "$RUN_DIR/logs"
torchrun \
    --nnodes="$NODE_COUNT" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="${MASTER_PORT:-6000}" \
    --nproc-per-node="$PROC_PER_NODE" \
    -m xtuner.v1.train.cli.sft \
    --config="${1:-$SCRIPT_DIR/speedtest_qwen35_35b_a3b.py}" \
    2>&1 | tee -a "$RUN_DIR/logs/node_$NODE_RANK.log"
