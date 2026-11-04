#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export JOB_NAME="${JOB_NAME:-zhutong-qwen35-speed}"
export NAMESPACE=ailab-llmshare
export CHARGED_GROUP=llmshare_temp
export IMAGE=registry.h.pjlab.org.cn/ailab-llmrazor/xtuner:pt29_20260904_76e7051
export RUN_DIR="${RUN_DIR:-$SOURCE_ROOT/work_dirs/qwen3p5_40b_a3b_speedtest/$JOB_NAME}"
export DRY_RUN="${DRY_RUN:-false}"

rjob submit \
    --name="$JOB_NAME" \
    --namespace="$NAMESPACE" \
    --charged-group="$CHARGED_GROUP" \
    --task-type=normal \
    --priority=5 \
    --private-machine=group \
    -P=4 --gpu=8 --cpu=64 --memory=524288 \
    --restart-policy=never \
    --image="$IMAGE" \
    --host-network=true \
    --share-host-shm=True \
    --use-file-store=False \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --custom-resources brainpp.cn/fuse=1 \
    --store-host-nvme \
    --mount=gpfs://gpfs1/llmrazor-share:/mnt/shared-storage-user/llmrazor-share \
    --mount=gpfs://gpfs1/linzhouhan:/mnt/shared-storage-user/linzhouhan \
    --mount=gpfs://gpfs2/plm-gpfs:/mnt/shared-storage-gpfs2/plm-gpfs \
    --dry-run="$DRY_RUN" \
    -e DISTRIBUTED_JOB=true \
    -e "RUN_DIR=$RUN_DIR" \
    -- bash "$SCRIPT_DIR/qwen35_speedtest_entrypoint.sh" \
        "$SCRIPT_DIR/speedtest_qwen35_35b_a3b.py"
