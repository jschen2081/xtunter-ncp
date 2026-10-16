#!/usr/bin/env bash
set -Eeuo pipefail
ALIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALIGNMENT_RUN_DIR="${ALIGNMENT_RUN_DIR:?Use the same directory as the successful Megatron capture}"
# Use the Python environment for the existing xTuner SP1 training test.
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  "$ALIGN_DIR/replay_xtuner.py" --run-dir "$ALIGNMENT_RUN_DIR" "$@"
