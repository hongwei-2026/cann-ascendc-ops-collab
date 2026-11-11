#!/bin/bash
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(git -C "$CODE" rev-parse --show-toplevel)"
SEAT=people/hongwei-2026/ConfusionTransposeD_2323/workdir/code
PROBLEM=/mnt/workspace/problems/ConfusionTransposeD_2323
mkdir -p "$PROBLEM/explorations" /mnt/workspace/exp_logs/ctd2323
RUN="$(mktemp -d "$PROBLEM/explorations/EXP_C007_$(date +%Y%m%d_%H%M%S)_XXXX")"
LOG_DIR="$RUN/logs"
mkdir -p "$LOG_DIR/source"
finish() {
    result=$?
    trap - EXIT
    set +e
    printf '%s\n' "$result" > "$LOG_DIR/exit_code.txt"
    python3 -B "$CODE/tests/p007_report.py" "$LOG_DIR" > "$LOG_DIR/terminal_report.log" 2>&1
    cat "$LOG_DIR/terminal_report.log"
    tar -czf "$RUN.tar.gz" -C "$RUN" logs
    cp "$RUN.tar.gz" /mnt/workspace/exp_logs/ctd2323/
    DOWNLOAD_DIR="$REPO/people/hongwei-2026/ConfusionTransposeD_2323/deliverables/cloud-logs"
    mkdir -p "$DOWNLOAD_DIR"
    cp "$RUN.tar.gz" "$DOWNLOAD_DIR/"
    printf '\nC007 exit=%s; archive=%s.tar.gz\n' "$result" "$RUN"
    printf 'CodeArts file explorer download: %s/%s.tar.gz\n' "$DOWNLOAD_DIR" "$(basename "$RUN")"
    printf 'If download is unavailable, copy C007_EVIDENCE_BEGIN through C007_EVIDENCE_END.\n'
    exit "$result"
}
trap finish EXIT
: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=1
export CTD_PERF_WARMUP=0 CTD_PERF_REPEAT=1
git -C "$REPO" rev-parse HEAD > "$LOG_DIR/candidate_commit.txt"
git -C "$REPO" rev-parse 4761239 > "$LOG_DIR/baseline_commit.txt"
git -C "$REPO" status --short -- "$SEAT" > "$LOG_DIR/source_status.log"
if [ -s "$LOG_DIR/source_status.log" ]; then
    printf 'Working tree changes in this seat; committed sources required.\n'
    exit 2
fi
{
    date -Is
    uname -a
    printf 'ASCEND_HOME_PATH=%s\n' "$ASCEND_HOME_PATH"
    for path in "$ASCEND_HOME_PATH/version.cfg" "$ASCEND_HOME_PATH/version.info"; do
        if [ -f "$path" ]; then cat "$path"; fi
    done
    npu-smi info || true
} > "$LOG_DIR/environment.log" 2>&1
mkdir -p "$RUN/baseline" "$RUN/candidate"
git -C "$REPO" archive 4761239 "$SEAT" | tar -x -C "$RUN/baseline"
git -C "$REPO" archive HEAD "$SEAT" | tar -x -C "$RUN/candidate"
for label in baseline candidate; do
    CURRENT="$RUN/$label/$SEAT"
    mkdir -p "$LOG_DIR/source/$label"
    cp -a "$CURRENT/op_host" "$CURRENT/op_kernel" "$LOG_DIR/source/$label/"
done
CANDIDATE="$RUN/candidate/$SEAT"
cp -a "$CANDIDATE/tests" "$LOG_DIR/source/tests"
cp "$CANDIDATE/examples/test_aclnn_confusion_transpose_d.cpp" "$LOG_DIR/source/common_sample.cpp"
cp "$CANDIDATE/examples/test_aclnn_confusion_transpose_d.cpp" "$RUN/baseline/$SEAT/examples/"
python3 - "$LOG_DIR/source" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
records = [{'file': str(path.relative_to(root)), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
           for path in sorted(root.rglob('*')) if path.is_file()]
(root.parent / 'source_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
PY
python3 -B "$CANDIDATE/tests/local_semantic_check.py" > "$LOG_DIR/local_semantic_check.log" 2>&1
python3 -B "$CANDIDATE/tests/p004_semantic_check.py" > "$LOG_DIR/p004_semantic_check.log" 2>&1
python3 -B "$CANDIDATE/tests/p007_semantic_check.py" > "$LOG_DIR/p007_semantic_check.log" 2>&1
PERF_STATUS=0
for label in baseline candidate; do
    CURRENT="$RUN/$label/$SEAT"
    printf '%s build.sh -e\n' "$label" | tee "$LOG_DIR/stage.txt"
    (cd "$CURRENT"; DEVICE_ID="${DEVICE_ID:-0}" bash build.sh -j4 -e) > "$LOG_DIR/${label}_build_e.log" 2>&1
    printf '%s 39-case bitwise regression\n' "$label" | tee "$LOG_DIR/stage.txt"
    python3 -B "$CANDIDATE/tests/p006_regression.py" --code-root "$CURRENT" \
        --directory "$LOG_DIR/${label}_regression" --device "${DEVICE_ID:-0}" \
        > "$LOG_DIR/${label}_regression.log" 2>&1
    printf '%s msprof kernel times\n' "$label" | tee "$LOG_DIR/stage.txt"
    python3 -B "$CANDIDATE/tests/p004_profile.py" --code-root "$CURRENT" \
        --cases "$LOG_DIR/${label}_regression" --output "$LOG_DIR/${label}_profile" \
        --device "${DEVICE_ID:-0}" --warmup 5 --repeat 20 \
        --probes rank2_swap_f32 baseline_tf_f32 dma_rows128_f32 larger_4d_f16 \
        > "$LOG_DIR/${label}_profile.log" 2>&1 || PERF_STATUS=3
done
printf 'completed; inspect functional results and profile separately\n' > "$LOG_DIR/stage.txt"
exit "$PERF_STATUS"
