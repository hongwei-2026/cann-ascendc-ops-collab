#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${ASCEND_HOME_PATH:?source the CANN set_env.sh first}"
mkdir -p "$ROOT/logs"
LOG_DIR="$(mktemp -d "$ROOT/logs/EXP_M009_$(date +%Y%m%d_%H%M%S)_XXXX")"
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$LOG_DIR/exit_code.txt"
    tar -czf "$LOG_DIR.tar.gz" -C "$(dirname "$LOG_DIR")" "$(basename "$LOG_DIR")"
    printf 'M009 logs: %s.tar.gz\n' "$LOG_DIR"
    exit "$result"
}
trap finish EXIT
exec > >(tee "$LOG_DIR/run_full.log") 2>&1
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('M009_SOURCE.json').read_text())
for name, expected in manifest['files_sha256_lf'].items():
    actual = hashlib.sha256(Path(name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    print(name, actual)
    if actual != expected:
        raise SystemExit('Source mismatch: ' + name)
print('source verification: PASS label=', manifest.get('label','?'))
print('Expect multi-core workspace >= 16MB (M010).')
PY
mkdir -p "$LOG_DIR/source/op_host" "$LOG_DIR/source/op_kernel"
cp M009_SOURCE.json "$LOG_DIR/SOURCE.json"
cp op_host/multilabel_margin_loss_tiling.cpp "$LOG_DIR/source/op_host/"
cp op_kernel/multilabel_margin_loss* "$LOG_DIR/source/op_kernel/"
python3 -B tests/local_semantic_check.py
DEVICE_ID="${DEVICE_ID:-0}" bash build.sh -j4 -e 2>&1 | tee "$LOG_DIR/build_e_full.log"
if [ -f build/op_kernel/multilabel_margin_loss.h ]; then
    cmp op_kernel/multilabel_margin_loss.h build/op_kernel/multilabel_margin_loss.h
    echo 'Build/source kernel bytes: MATCH'
else
    echo 'Build copy not found at build/op_kernel; source snapshot is archived.'
fi
python3 -B tests/npu_regression.py --directory "$LOG_DIR/regression" --device "${DEVICE_ID:-0}"
