#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -n 65536

# Training defaults exported to the Python configuration; caller values take precedence.
export ZERO_STAGE="${ZERO_STAGE:-2}"
# FSDP gradient reduction precision; parameter computation remains BF16.
export REDUCE_DTYPE="${REDUCE_DTYPE:-fp32}"
export GRADIENT_ACCUMULATION_NO_SYNC="${GRADIENT_ACCUMULATION_NO_SYNC:-1}"
export ENABLE_COMPILE="${ENABLE_COMPILE:-1}"
export RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0}"
export ENABLE_TRACE="${ENABLE_TRACE:-0}"
# Tokens per LM-head cross-entropy chunk; override via the environment.
export LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-512}"

export SP_SIZE="${SP_SIZE:-1}"
export TRAIN_ITERS="${TRAIN_ITERS:-100}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export SEED="${SEED:-1234}"
export LR="${LR:-8e-4}"
export MIN_LR="${MIN_LR:-8e-5}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2000}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-143052}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-14306}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-wsd}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE=/mnt/shared-storage-user/llmrazor-share/heqiushi
DATA_PATH="${DATA_PATH:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/dolma3_mix-150B-1025/megatron_qwen35_8192/indexed}"
CACHE="${DATA_CACHE_ROOT:-$BASE/xtuner_text_data_cache}"
OUT="${WORK_DIR:-$BASE/xtuner_train_runs/qwen35_gbs128_zero2_fa3_nosync_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
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
resolve_run_dir "$OUT" "$@"
OUT="$run_dir"
set -- "${launch_args[@]}"
PY="${XTUNER_PYTHON:-/usr/bin/python}"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
export PYTHONPATH="$BASE/xtuner_text_runtime_deps:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

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
# Validate before third-party imports.
"$PY" "$REPO/examples/v1/scripts/megatron_alignment/training_environment.py"
"$PY" "$REPO/examples/v1/scripts/megatron_alignment/runtime_requirements.py"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$(basename "$OUT")/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/$(basename "$OUT")/inductor}"
cd "$REPO"
MANIFEST="${TEXT_MANIFEST:-}"
FORMAT="${DATA_FORMAT:-indexed}"
if [[ "$FORMAT" == text && -z "$MANIFEST" ]]; then
  MANIFEST=$("$PY" examples/v1/scripts/megatron_alignment/prepare_qwen35_text_data.py \
  --input "$DATA_PATH" --cache-root "$CACHE/normalized_text")
fi
finish() {
  rc=$?
  trap - EXIT
  if [[ -d "$OUT" ]]; then
    echo "$rc" > "$OUT/exit_code"
    date --iso-8601=seconds > "$OUT/finished_at.txt"
    if (( rc != 0 )); then echo FAILED > "$OUT/status"; fi
  fi
  exit "$rc"
}
trap finish EXIT
echo "CONFIG: reduce_dtype=$REDUCE_DTYPE FA3=$XTUNER_USE_FA3 ZeRO=$ZERO_STAGE no_sync=$GRADIENT_ACCUMULATION_NO_SYNC compile=$ENABLE_COMPILE recompute_ratio=$RECOMPUTE_RATIO trace=$ENABLE_TRACE lm_head_chunk_size=$LOSS_CHUNK_SIZE"
# Explicit CLI arguments override the environment defaults.
echo "OUTPUT=$OUT"
"$PY" examples/v1/scripts/megatron_alignment/train_qwen35.py \
  --run-dir "$OUT" --data-format "$FORMAT" --data-path "$DATA_PATH" \
  --text-manifest "$MANIFEST" --data-cache "$CACHE/tokenization" \
  "$@"

"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  -m xtuner.v1.train.cli.sft --config "$OUT/trainer_config.py"
"$PY" examples/v1/scripts/megatron_alignment/train_qwen35.py --run-dir "$OUT" --collect-metrics
