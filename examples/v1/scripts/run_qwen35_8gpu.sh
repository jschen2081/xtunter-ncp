#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE=/mnt/shared-storage-user/llmrazor-share/heqiushi
DATA_PATH="${DATA_PATH:-/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/dolma3_mix-150B-1025/megatron_qwen35_8192/indexed}"
CACHE="${DATA_CACHE_ROOT:-$BASE/xtuner_text_data_cache}"
OUT="${WORK_DIR:-$BASE/xtuner_train_runs/qwen35_gbs128_zero2_fa3_nosync_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
PY="${XTUNER_PYTHON:-/usr/bin/python}"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
export PYTHONPATH="$BASE/xtuner_text_runtime_deps:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XTUNER_USE_FA3=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
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
echo "OUTPUT=$OUT"
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  examples/v1/scripts/megatron_alignment/train_qwen35.py \
  --run-dir "$OUT" --data-format "$FORMAT" --data-path "$DATA_PATH" \
  --text-manifest "$MANIFEST" --data-cache "$CACHE/tokenization" \
  --zero-stage 2 --gradient-accumulation-no-sync --compile --recompute-ratio 0 \
  --steps "${TRAIN_ITERS:-100}" --global-batch-size "${GLOBAL_BATCH_SIZE:-128}" \
  --sequence-length "${SEQ_LENGTH:-8192}" --seed "${SEED:-1234}" \
  --lr "${LR:-8e-4}" --min-lr "${MIN_LR:-8e-5}" \
  --warmup-iters "${LR_WARMUP_ITERS:-2000}" --decay-iters "${LR_DECAY_ITERS:-143052}" \
  --wsd-decay-iters "${LR_WSD_DECAY_ITERS:-14306}" "$@"
