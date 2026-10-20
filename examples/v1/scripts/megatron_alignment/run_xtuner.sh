#!/usr/bin/env bash
set -Eeuo pipefail

# Editable training settings: use 1 to enable and 0 to disable.
USE_FA3=1
ZERO_STAGE=2
GRADIENT_ACCUMULATION_NO_SYNC=1
ENABLE_COMPILE=1
RECOMPUTE_RATIO=0
ENABLE_TRACE=0

ALIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALIGNMENT_RUN_DIR="${ALIGNMENT_RUN_DIR:?Use the same directory as the successful Megatron capture}"
export XTUNER_USE_FA3="$USE_FA3"
runtime_args=(--zero-stage "$ZERO_STAGE" --recompute-ratio "$RECOMPUTE_RATIO")
if (( GRADIENT_ACCUMULATION_NO_SYNC )); then
  runtime_args+=(--gradient-accumulation-no-sync)
else
  runtime_args+=(--no-gradient-accumulation-no-sync)
fi
if (( ENABLE_COMPILE )); then
  runtime_args+=(--compile)
else
  runtime_args+=(--no-compile)
fi
if (( ENABLE_TRACE )); then
  runtime_args+=(--profile-trace)
else
  runtime_args+=(--no-profile-trace)
fi
echo "CONFIG: FA3=$USE_FA3 ZeRO=$ZERO_STAGE no_sync=$GRADIENT_ACCUMULATION_NO_SYNC compile=$ENABLE_COMPILE recompute_ratio=$RECOMPUTE_RATIO trace=$ENABLE_TRACE"
# Kernel selection must precede package imports in the standard CLI.

PY="${XTUNER_PYTHON:-python}"
REPO="$(cd "$ALIGN_DIR/../../../.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
kernel_exports="$("$PY" "$ALIGN_DIR/kernel_config.py")"
while IFS='=' read -r kernel_name kernel_value; do
  export "$kernel_name=$kernel_value"
done <<< "$kernel_exports"
unset XTUNER_HF_IMPL
"$PY" "$ALIGN_DIR/runtime_requirements.py"
# Select the output directory for the shared native Trainer entry.
sp_size=1
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
  case "${arguments[i]}" in
    --sp-size) sp_size="${arguments[i+1]:-}" ;;
    --sp-size=*) sp_size="${arguments[i]#*=}" ;;
  esac
done
"$PY" "$ALIGN_DIR/replay_xtuner.py" --run-dir "$ALIGNMENT_RUN_DIR" "$@" "${runtime_args[@]}"
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  -m xtuner.v1.train.cli.sft --config "$ALIGNMENT_RUN_DIR/xtuner_sp${sp_size}_native/trainer_config.py"
"$PY" "$ALIGN_DIR/replay_xtuner.py" --run-dir "$ALIGNMENT_RUN_DIR" --sp-size "$sp_size" --collect-metrics
