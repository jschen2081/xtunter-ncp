#!/usr/bin/env bash
set -Eeuo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$A/../../../.." && pwd)"
OUT="${1:?Pass prepared output directory}"
shift
KEEP=/mnt/shared-storage-user/llmrazor-share/heqiushi/gpu_keepalive/gpu_keepalive.sh
export PATH="/usr/bin:$PATH" PYTHONPATH="$R" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XTUNER_PYTHON="${XTUNER_PYTHON:-/usr/bin/python}"
"$XTUNER_PYTHON" "$A/runtime_requirements.py"
exec 9>"$OUT/runner.lock"
flock -n 9 || exit 2
[[ ! -e "$OUT/xtuner_sp1.jsonl" ]] || exit 2
finish() {
    rc=$?
    trap - EXIT
    echo "$rc" > "$OUT/exit_code"
    date --iso-8601=seconds > "$OUT/finished_at.txt"
    if (( rc == 0 )); then echo COMPLETED > "$OUT/status"; else echo FAILED > "$OUT/status"; fi
    flock -u 9
    exec 9>&-
    if ! bash "$KEEP" start >> "$OUT/keepalive.log" 2>&1; then echo KEEPALIVE_RESTORE_FAILED >> "$OUT/status"; fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date --iso-8601=seconds > "$OUT/started_at.txt"
echo STOPPING_KEEPALIVE > "$OUT/status"
bash "$KEEP" stop >> "$OUT/keepalive.log" 2>&1
for ((i=0; i<60; i++)); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]] && break
    sleep 1
done
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || { echo 'GPU still busy'; exit 1; }
cd "$R"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$(basename "$OUT")/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/$(basename "$OUT")/inductor}"
git rev-parse HEAD > "$OUT/xtuner_commit.txt"
git diff HEAD --binary > "$OUT/xtuner_source.patch"
cp "$A/replay_xtuner.py" "$OUT/replay_xtuner.executed.py"
echo RUNNING > "$OUT/status"
# Training settings are edited in run_xtuner.sh.
ALIGNMENT_RUN_DIR="$OUT" bash "$A/run_xtuner.sh" "$@"
echo 0 > "$OUT/training_exit_code"
echo COMPARING > "$OUT/status"
if [[ -f "$OUT/conversion_manifest.json" ]]; then
    "$XTUNER_PYTHON" "$A/compare_converted_capture.py" "$OUT" > "$OUT/comparison.log" 2>&1
else
    "$XTUNER_PYTHON" "$A/compare_results.py" "$OUT" > "$OUT/comparison.log" 2>&1
fi
