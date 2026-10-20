#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -n 65536

# Editable training settings: use 1 to enable and 0 to disable.
USE_FA3=1
ZERO_STAGE=2
GRADIENT_ACCUMULATION_NO_SYNC=1
ENABLE_COMPILE=1
RECOMPUTE_RATIO=0
ENABLE_TRACE=0

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE=/mnt/shared-storage-user/llmrazor-share/heqiushi
DATA_PATH="${DATA_PATH:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/dolma3_mix-150B-1025/megatron_qwen35_8192/indexed}"
CACHE="${DATA_CACHE_ROOT:-$BASE/xtuner_text_data_cache}"
OUT="${WORK_DIR:-$BASE/xtuner_train_runs/qwen35_gbs128_zero2_fa3_nosync_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
PY="${XTUNER_PYTHON:-/usr/bin/python}"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
export PYTHONPATH="$BASE/xtuner_text_runtime_deps:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XTUNER_USE_FA3="$USE_FA3" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XTUNER_DETERMINISTIC=true XTUNER_GDN_CUSTOM_OP=true XTUNER_GDN_FUSED_QK_L2=true
export XTUNER_GDN_FUSED_NORM_GATE=true NCP_Q35V2_TORCH_RMS=1 NCP_Q35V2_MUDD_RMS=1
export NCP_Q35V2_TE_RMS=0 NCP_Q35V2_GDN_TORCH_INPUT_NORM=1 NCP_Q35V2_TRITON_DUAL_CROSS_MUDD=0
unset XTUNER_HF_IMPL
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
# Apply the settings above after optional arguments.
echo "OUTPUT=$OUT"
"$PY" examples/v1/scripts/megatron_alignment/train_qwen35.py \
  --run-dir "$OUT" --data-format "$FORMAT" --data-path "$DATA_PATH" \
  --text-manifest "$MANIFEST" --data-cache "$CACHE/tokenization" \
  --sp-size "${SP_SIZE:-1}" \
  --steps "${TRAIN_ITERS:-100}" --global-batch-size "${GLOBAL_BATCH_SIZE:-128}" \
  --sequence-length "${SEQ_LENGTH:-8192}" --seed "${SEED:-1234}" \
  --lr "${LR:-8e-4}" --min-lr "${MIN_LR:-8e-5}" \
  --warmup-iters "${LR_WARMUP_ITERS:-2000}" --decay-iters "${LR_DECAY_ITERS:-143052}" \
  --wsd-decay-iters "${LR_WSD_DECAY_ITERS:-14306}" --lr-decay-style "${LR_DECAY_STYLE:-wsd}" "$@" \
  "${runtime_args[@]}"

"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  -m xtuner.v1.train.cli.sft --config "$OUT/trainer_config.py"
"$PY" examples/v1/scripts/megatron_alignment/train_qwen35.py --run-dir "$OUT" --collect-metrics
