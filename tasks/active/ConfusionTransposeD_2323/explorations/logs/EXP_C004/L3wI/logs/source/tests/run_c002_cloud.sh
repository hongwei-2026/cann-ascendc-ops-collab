#!/bin/bash
# EXP_C002 cloud/NPU harness: sample -e + UB-safe generalization regression.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${ASCEND_HOME_PATH:?source the CANN set_env.sh first}"
mkdir -p "$ROOT/logs"
LOG_DIR="$(mktemp -d "$ROOT/logs/EXP_C002_$(date +%Y%m%d_%H%M%S)_XXXX")"
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$LOG_DIR/exit_code.txt"
    tar -czf "$LOG_DIR.tar.gz" -C "$(dirname "$LOG_DIR")" "$(basename "$LOG_DIR")"
    printf 'C002 logs: %s.tar.gz\n' "$LOG_DIR"
    exit "$result"
}
trap finish EXIT
exec > >(tee "$LOG_DIR/run_full.log") 2>&1

mkdir -p "$LOG_DIR/source/op_host" "$LOG_DIR/source/op_kernel" "$LOG_DIR/source/examples" "$LOG_DIR/source/tests"
cp -a op_host/. "$LOG_DIR/source/op_host/"
cp -a op_kernel/. "$LOG_DIR/source/op_kernel/"
cp examples/test_aclnn_confusion_transpose_d.cpp "$LOG_DIR/source/examples/"
cp tests/npu_regression.py tests/local_semantic_check.py tests/run_c002_cloud.sh "$LOG_DIR/source/tests/"

python3 -B tests/local_semantic_check.py
DEVICE_ID="${DEVICE_ID:-0}" bash build.sh -j4 -e 2>&1 | tee "$LOG_DIR/build_e_full.log"
python3 -B tests/npu_regression.py --directory "$LOG_DIR/regression" --device "${DEVICE_ID:-0}"
