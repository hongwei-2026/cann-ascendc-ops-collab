#!/bin/bash
# P004 only. Isolated A/B build, precision and msprof; archives failures too.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(git -C "$CODE" rev-parse --show-toplevel)"
SEAT=people/hongwei-2026/ConfusionTransposeD_2323/workdir/code
BASE_REF=545f8d2
PROBLEM=/mnt/workspace/problems/ConfusionTransposeD_2323
mkdir -p "$PROBLEM/explorations" /mnt/workspace/exp_logs/ctd2323
RUN="$(mktemp -d "$PROBLEM/explorations/EXP_C004_$(date +%Y%m%d_%H%M%S)_XXXX")"
LOG_DIR="$RUN/logs"
mkdir -p "$LOG_DIR"
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$LOG_DIR/exit_code.txt"
    tar -czf "$RUN.tar.gz" -C "$RUN" logs
    cp "$RUN.tar.gz" /mnt/workspace/exp_logs/ctd2323/
    printf '\nC004 exit=%s; download: %s.tar.gz\n' "$result" "$RUN"
    printf 'Also available in /mnt/workspace/exp_logs/ctd2323/\n'
    exit "$result"
}
trap finish EXIT
exec > >(tee "$LOG_DIR/run_full.log") 2>&1
: "${ASCEND_HOME_PATH:?source the CANN set_env.sh first}"

git -C "$REPO" rev-parse HEAD > "$LOG_DIR/candidate_commit.txt"
git -C "$REPO" rev-parse "$BASE_REF" > "$LOG_DIR/baseline_commit.txt"
git -C "$REPO" status --short -- "$SEAT" > "$LOG_DIR/source_status.txt"
if [ -s "$LOG_DIR/source_status.txt" ]; then
    echo 'Commit or stash local edits in this seat before running; archive uses committed sources.'
    exit 2
fi
{
    date -Is
    uname -a
    printf 'ASCEND_HOME_PATH=%s\n' "$ASCEND_HOME_PATH"
    for path in "$ASCEND_HOME_PATH/version.cfg" "$ASCEND_HOME_PATH/version.info"; do
        if [ -f "$path" ]; then cat "$path"; fi
    done
    command -v msprof || true
    npu-smi info || true
} > "$LOG_DIR/environment.log" 2>&1
mkdir -p "$RUN/baseline" "$RUN/candidate"
git -C "$REPO" archive "$BASE_REF" "$SEAT" | tar -x -C "$RUN/baseline"
git -C "$REPO" archive HEAD "$SEAT" | tar -x -C "$RUN/candidate"
BASELINE="$RUN/baseline/$SEAT"
CANDIDATE="$RUN/candidate/$SEAT"
# Same profiling driver for both; baseline Host/Kernel stay exactly at BASE_REF.
cp "$CANDIDATE/examples/test_aclnn_confusion_transpose_d.cpp" "$BASELINE/examples/"
for label in baseline candidate; do
    mkdir -p "$LOG_DIR/source/$label"
    cp -a "$RUN/$label/$SEAT/op_host" "$RUN/$label/$SEAT/op_kernel" "$LOG_DIR/source/$label/"
done
cp -a "$CANDIDATE/tests" "$LOG_DIR/source/tests"
cp "$CANDIDATE/examples/test_aclnn_confusion_transpose_d.cpp" "$LOG_DIR/source/common_sample.cpp"
python3 -B "$CANDIDATE/tests/local_semantic_check.py" | tee "$LOG_DIR/local_semantic.log"
python3 -B "$CANDIDATE/tests/p004_semantic_check.py" | tee "$LOG_DIR/p004_semantic.log"

PERF_STATUS=0
for label in baseline candidate; do
    CURRENT="$RUN/$label/$SEAT"
    (cd "$CURRENT"; DEVICE_ID="${DEVICE_ID:-0}" bash build.sh -j4 -e) 2>&1 | tee "$LOG_DIR/${label}_build_e_full.log"
    python3 -B "$CANDIDATE/tests/npu_regression.py" --include-p004 --bitwise \
        --code-root "$CURRENT" --binary "$CURRENT/examples/build/sample" \
        --directory "$LOG_DIR/${label}_regression" --device "${DEVICE_ID:-0}" \
        2>&1 | tee "$LOG_DIR/${label}_regression_full.log"
    if python3 -B "$CANDIDATE/tests/p004_profile.py" --code-root "$CURRENT" \
        --cases "$LOG_DIR/${label}_regression" --output "$LOG_DIR/${label}_profile" \
        --device "${DEVICE_ID:-0}" 2>&1 | tee "$LOG_DIR/${label}_profile_full.log"; then
        :
    else
        PERF_STATUS=3
    fi
done
python3 - "$LOG_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
baseline = {r.get('case'): r for r in json.loads((root / 'baseline_profile/profile_summary.json').read_text())}
candidate = json.loads((root / 'candidate_profile/profile_summary.json').read_text())
comparison = []
for row in candidate:
    old = baseline.get(row.get('case'), {})
    if old.get('result') == row.get('result') == 'PROFILED':
        comparison.append({'case': row['case'], 'baseline_us': old['median_us'],
                           'candidate_us': row['median_us'], 'speedup': old['median_us'] / row['median_us'],
                           'scope': 'same-machine custom kernel A/B; not hidden platform cases or TBE'})
(root / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
print(json.dumps(comparison, indent=2))
PY
exit "$PERF_STATUS"
