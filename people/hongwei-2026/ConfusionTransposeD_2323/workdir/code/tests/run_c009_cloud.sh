#!/usr/bin/env bash
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(git -C "$CODE" rev-parse --show-toplevel)"
SEAT=people/hongwei-2026/ConfusionTransposeD_2323/workdir/code
PROBLEM=/mnt/workspace/problems/ConfusionTransposeD_2323
mkdir -p "$PROBLEM/explorations" /mnt/workspace/exp_logs/ctd2323
RUN="$(mktemp -d "$PROBLEM/explorations/EXP_C009_$(date +%Y%m%d_%H%M%S)_XXXX")"
LOG="$RUN/logs"
mkdir -p "$LOG"
finish() {
    result=$?
    trap - EXIT
    set +e
    printf '%s\n' "$result" > "$LOG/exit_code.txt"
    printf '\nC009_EVIDENCE_BEGIN\nexit=%s\n' "$result"
    for file in candidate_commit.txt stage.txt baseline_regression.log candidate_regression.log candidate_boundary.log probe_check.log; do
        if [ -f "$LOG/$file" ]; then printf '\n--- %s ---\n' "$file"; cat "$LOG/$file"; fi
    done
    for file in baseline_build_e.log candidate_build_e.log; do
        if [ -f "$LOG/$file" ]; then printf '\n--- %s (last 60 lines) ---\n' "$file"; tail -n 60 "$LOG/$file"; fi
    done
    tar -czf "$RUN.tar.gz" -C "$RUN" logs
    cp "$RUN.tar.gz" /mnt/workspace/exp_logs/ctd2323/
    printf '\narchive=%s.tar.gz\nC009_EVIDENCE_END\n' "$RUN"
    exit "$result"
}
trap finish EXIT
: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=1 CTD_C005_LOG_LEVEL=3
git -C "$REPO" rev-parse HEAD > "$LOG/candidate_commit.txt"
git -C "$REPO" status --short -- "$SEAT" > "$LOG/source_status.log"
if [ -s "$LOG/source_status.log" ]; then
    echo 'Changes in operator seat; committed source required.' > "$LOG/stage.txt"
    exit 2
fi
for label in baseline candidate; do
    mkdir -p "$RUN/$label"
    ref=HEAD
    if [ "$label" = baseline ]; then ref=4761239; fi
    git -C "$REPO" archive "$ref" "$SEAT" | tar -x -C "$RUN/$label"
done
CANDIDATE="$RUN/candidate/$SEAT"
cp "$CANDIDATE/examples/test_aclnn_confusion_transpose_d.cpp" "$RUN/baseline/$SEAT/examples/"
cp -a "$CANDIDATE/op_host" "$CANDIDATE/op_kernel" "$LOG/"
printf 'local semantic\n' > "$LOG/stage.txt"
python3 -B "$CANDIDATE/tests/local_semantic_check.py" > "$LOG/local_semantic.log" 2>&1
for label in baseline candidate; do
    CURRENT="$RUN/$label/$SEAT"
    printf '%s build.sh -e\n' "$label" > "$LOG/stage.txt"
    (cd "$CURRENT"; DEVICE_ID="${DEVICE_ID:-0}" bash build.sh -j4 -e) > "$LOG/${label}_build_e.log" 2>&1
    printf '%s 39-case bitwise regression\n' "$label" > "$LOG/stage.txt"
    python3 -B "$CANDIDATE/tests/p006_regression.py" --code-root "$CURRENT" \
        --directory "$LOG/${label}_regression" --device "${DEVICE_ID:-0}" > "$LOG/${label}_regression.log" 2>&1
done
printf 'candidate boundary diagnostics\n' > "$LOG/stage.txt"
python3 -B "$CANDIDATE/tests/c005_diagnostic.py" --version candidate --code-root "$CANDIDATE" \
    --directory "$LOG/candidate_boundary" --device "${DEVICE_ID:-0}" > "$LOG/candidate_boundary.log" 2>&1
printf 'probe evidence check\n' > "$LOG/stage.txt"
python3 -B "$CANDIDATE/tests/c009_probe_report.py" "$LOG" > "$LOG/probe_check.log" 2>&1
printf 'completed; platform visibility is unverified\n' > "$LOG/stage.txt"
