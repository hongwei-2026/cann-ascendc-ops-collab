#!/bin/bash
# P005 diagnostic only. Reuses exact C004 builds; never changes Host/Kernel.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(git -C "$CODE" rev-parse --show-toplevel)"
SEAT=people/hongwei-2026/ConfusionTransposeD_2323/workdir/code
PROBLEM=/mnt/workspace/problems/ConfusionTransposeD_2323
C004_RUN="${CTD_C004_RUN:-$PROBLEM/explorations/EXP_C004_20261110_142548_L3wI}"
mkdir -p "$PROBLEM/explorations" /mnt/workspace/exp_logs/ctd2323
RUN="$(mktemp -d "$PROBLEM/explorations/EXP_C005_$(date +%Y%m%d_%H%M%S)_XXXX")"
LOG_DIR="$RUN/logs"
mkdir -p "$LOG_DIR/source"
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$LOG_DIR/diagnostic_exit_code.txt"
    tar -czf "$RUN.tar.gz" -C "$RUN" logs
    cp "$RUN.tar.gz" /mnt/workspace/exp_logs/ctd2323/
    printf '\nC005 diagnostic exit=%s (not kernel acceptance); download: %s.tar.gz\n' "$result" "$RUN"
    exit "$result"
}
trap finish EXIT
exec > >(tee "$LOG_DIR/run_full.log") 2>&1
: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"
git -C "$REPO" rev-parse HEAD > "$LOG_DIR/diagnostic_commit.txt"
git -C "$REPO" status --short -- "$SEAT" > "$LOG_DIR/source_status.txt"
if [ -s "$LOG_DIR/source_status.txt" ]; then
    echo 'Local changes in this operator seat; pull the committed diagnostic before running.'
    exit 2
fi
for label in baseline candidate; do
    CURRENT="$C004_RUN/$label/$SEAT"
    if [ ! -f "$CURRENT/build/packages/vendors/confusion_transpose_d_custom/op_api/lib/libcust_opapi.so" ]; then
        echo "C004 built package missing: $CURRENT; set CTD_C004_RUN to the existing C004 run folder."
        exit 2
    fi
done
python3 - "$REPO" "$C004_RUN" "$SEAT" "$LOG_DIR" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
repo, run, seat, log = map(Path, sys.argv[1:])
records = []
for label, ref in [('baseline', '545f8d2'), ('candidate', '4761239')]:
    root = run / label / seat
    for section in ('op_host', 'op_kernel'):
        for p in (root / section).rglob('*'):
            if not p.is_file(): continue
            rel = p.relative_to(root)
            original = subprocess.check_output(['git', '-C', str(repo), 'show', f'{ref}:{seat.as_posix()}/{rel.as_posix()}'])
            if p.read_bytes().replace(b'\r\n', b'\n') != original.replace(b'\r\n', b'\n'):
                raise SystemExit(f'Source changed since C004: {p}')
            records.append({'version': label, 'file': str(rel), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    # C004 applied the same candidate example driver to both versions.
    driver = root / 'examples/test_aclnn_confusion_transpose_d.cpp'
    expected = subprocess.check_output(['git', '-C', str(repo), 'show', f'4761239:{seat.as_posix()}/examples/test_aclnn_confusion_transpose_d.cpp'])
    assert driver.read_bytes().replace(b'\r\n', b'\n') == expected.replace(b'\r\n', b'\n')
(log / 'source_integrity.json').write_text(json.dumps(records, indent=2) + '\n')
print('C004 Host/Kernel and shared driver source integrity verified; reusing built packages.')
PY
cp "$CODE/tests/c005_diagnostic.py" "$CODE/tests/npu_regression.py" "$CODE/tests/run_c005_diagnostic.sh" "$LOG_DIR/source/"
cp "$CODE/samples/reference.py" "$LOG_DIR/source/reference.py"
{
    date -Is
    printf 'ASCEND_HOME_PATH=%s\n' "$ASCEND_HOME_PATH"
    npu-smi info || true
} > "$LOG_DIR/environment.log" 2>&1
python3 - "$ASCEND_HOME_PATH" "$LOG_DIR" <<'PY'
import json, shutil, sys
from pathlib import Path
toolkit, log = map(Path, sys.argv[1:])
records = []
for base in [toolkit / 'include', toolkit / 'compiler/tikcpp', toolkit / 'aarch64-linux/ascendc']:
    if not base.is_dir(): continue
    for path in base.rglob('*.h'):
        lower = path.as_posix().lower()
        if 'gather' not in path.name.lower() and 'transpose' not in path.name.lower(): continue
        if any(v in lower for v in ['dav_c310', 'dav_c100', 'dav_c220_cube', 'dav_3510', 'dav_310r6']): continue
        if path.stat().st_size > 512 * 1024: continue
        dest = log / 'cann_api_headers' / path.relative_to(toolkit)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        records.append(str(path.relative_to(toolkit)))
(log / 'cann_api_headers.json').write_text(json.dumps(records, indent=2) + '\n')
print('Read-only CANN Gather/Transpose header snapshots:', len(records))
PY
python3 -B "$CODE/tests/local_semantic_check.py" | tee "$LOG_DIR/local_semantic.log"
for label in baseline candidate; do
    python3 -B "$CODE/tests/c005_diagnostic.py" --version "$label" \
        --code-root "$C004_RUN/$label/$SEAT" --directory "$LOG_DIR/$label" \
        --device "${DEVICE_ID:-0}" 2>&1 | tee "$LOG_DIR/${label}_full.log"
done
python3 - "$LOG_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
old = {r['case']: r for r in json.loads((root / 'baseline/results.json').read_text())}
new = json.loads((root / 'candidate/results.json').read_text())
comparison = []
for r in new:
    b = old[r['case']]
    pair = (b['result'], r['result'])
    if pair == ('PASS_BITWISE', 'PASS_BITWISE'): finding = 'BOTH_PASS'
    elif pair[0] == 'PASS_BITWISE' and pair[1] == 'FAIL_OR_REJECTED': finding = 'P004_REGRESSION_CANDIDATE'
    elif pair == ('FAIL_OR_REJECTED', 'FAIL_OR_REJECTED'): finding = 'BOTH_FAIL_OR_REJECT; INSPECT_ERROR_PHASE'
    else: finding = 'INSPECT_RAW_LOGS'
    comparison.append({'case': r['case'], 'category': r['category'], 'baseline': pair[0], 'candidate': pair[1], 'finding': finding})
(root / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
for r in comparison:
    if r['finding'] != 'BOTH_PASS': print(json.dumps(r))
print('Diagnostic completed. Results are probes, not platform case identities or acceptance PASS.')
PY
