#!/usr/bin/env bash
set -Eeuo pipefail

# Training defaults exported to the Python configuration; caller values take precedence.
export ZERO_STAGE="${ZERO_STAGE:-2}"
export GRADIENT_ACCUMULATION_NO_SYNC="${GRADIENT_ACCUMULATION_NO_SYNC:-1}"
export ENABLE_COMPILE="${ENABLE_COMPILE:-1}"
export RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0}"
export ENABLE_TRACE="${ENABLE_TRACE:-0}"
# Tokens per LM-head cross-entropy chunk; override via the environment.
export LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-512}"

ALIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve the output directory once, before chdir or any output-side effects.
resolve_run_dir() {
  run_dir="$1"
  shift
  launch_args=()
  while (( $# )); do
    case "$1" in
      --run-dir)
        if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
          echo "--run-dir requires a path" >&2
          return 2
        fi
        run_dir="$2"
        shift 2
        ;;
      --run-dir=*) run_dir="${1#*=}"; shift ;;
      *) launch_args+=("$1"); shift ;;
    esac
  done
  [[ -n "$run_dir" ]] || { echo "A run directory is required" >&2; return 2; }
  run_dir="$(realpath -m -- "$run_dir")"
}
resolve_run_dir "${ALIGNMENT_RUN_DIR:-}" "$@"
export ALIGNMENT_RUN_DIR="$run_dir"
set -- "${launch_args[@]}"
# Kernel selection must precede package imports in the standard CLI.

PY="${XTUNER_PYTHON:-python}"
REPO="$(cd "$ALIGN_DIR/../../../.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Kernel environment: caller values take precedence.
export XTUNER_USE_FA3="${XTUNER_USE_FA3:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export XTUNER_DETERMINISTIC="${XTUNER_DETERMINISTIC:-false}"
export NCP_Q35V2_CACHE_CROSS_SOURCE_RMS="${NCP_Q35V2_CACHE_CROSS_SOURCE_RMS:-1}"
export XTUNER_GDN_CUSTOM_OP="${XTUNER_GDN_CUSTOM_OP:-true}"
export XTUNER_GDN_FUSED_QK_L2="${XTUNER_GDN_FUSED_QK_L2:-true}"
export XTUNER_GDN_FUSED_NORM_GATE="${XTUNER_GDN_FUSED_NORM_GATE:-true}"
export NCP_Q35V2_TORCH_RMS="${NCP_Q35V2_TORCH_RMS:-1}"
export NCP_Q35V2_MUDD_RMS="${NCP_Q35V2_MUDD_RMS:-1}"
export NCP_Q35V2_GDN_TORCH_INPUT_NORM="${NCP_Q35V2_GDN_TORCH_INPUT_NORM:-1}"
export NCP_Q35V2_TRITON_DUAL_CROSS_MUDD="${NCP_Q35V2_TRITON_DUAL_CROSS_MUDD:-0}"

# These launchers use the native implementation.
unset XTUNER_HF_IMPL
echo "CONFIG: FA3=$XTUNER_USE_FA3 ZeRO=$ZERO_STAGE no_sync=$GRADIENT_ACCUMULATION_NO_SYNC compile=$ENABLE_COMPILE recompute_ratio=$RECOMPUTE_RATIO trace=$ENABLE_TRACE lm_head_chunk_size=$LOSS_CHUNK_SIZE"
# Validate before third-party imports.
"$PY" "$ALIGN_DIR/training_environment.py"
"$PY" "$ALIGN_DIR/runtime_requirements.py"
# Select the output directory for the shared native Trainer entry.
sp_size="${SP_SIZE:-1}"
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
  case "${arguments[i]}" in
    --sp-size) sp_size="${arguments[i+1]:-}" ;;
    --sp-size=*) sp_size="${arguments[i]#*=}" ;;
  esac
done
"$PY" "$ALIGN_DIR/replay_xtuner.py" --run-dir "$ALIGNMENT_RUN_DIR" "$@"
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  -m xtuner.v1.train.cli.sft --config "$ALIGNMENT_RUN_DIR/xtuner_sp${sp_size}_native/trainer_config.py"
"$PY" "$ALIGN_DIR/replay_xtuner.py" --run-dir "$ALIGNMENT_RUN_DIR" --sp-size "$sp_size" --collect-metrics
