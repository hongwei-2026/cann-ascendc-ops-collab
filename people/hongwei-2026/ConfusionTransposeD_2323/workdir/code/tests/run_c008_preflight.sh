#!/usr/bin/env bash
set -euo pipefail

CODE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(git -C "$CODE" rev-parse --show-toplevel)"
PROBLEM=/mnt/workspace/problems/ConfusionTransposeD_2323
OUT=/mnt/workspace/exp_logs/ctd2323
mkdir -p "$PROBLEM/explorations" "$OUT"
RUN="$(mktemp -d "$PROBLEM/explorations/EXP_C008_$(date +%Y%m%d_%H%M%S)_XXXX")"
LOG="$RUN/logs"
mkdir -p "$LOG"

finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$LOG/exit_code.txt"
    tar -czf "$RUN.tar.gz" -C "$RUN" logs
    cp "$RUN.tar.gz" "$OUT/"
    printf '\nC008_EVIDENCE_BEGIN\n'
    printf 'exit=%s\n' "$result"
    if [ -f "$LOG/environment.log" ]; then cat "$LOG/environment.log"; fi
    if [ -f "$LOG/tbe_file_hits.txt" ]; then
        printf 'TBE_FILE_HITS_BEGIN\n'
        cat "$LOG/tbe_file_hits.txt"
        printf 'TBE_FILE_HITS_END\n'
    fi
    if [ -f "$LOG/tbe_symbol_hits.txt" ]; then
        printf 'TBE_SYMBOL_HITS_BEGIN\n'
        cat "$LOG/tbe_symbol_hits.txt"
        printf 'TBE_SYMBOL_HITS_END\n'
    fi
    if [ -f "$LOG/transpose_api_hits.txt" ]; then
        printf 'TRANSPOSE_API_HITS_BEGIN\n'
        cat "$LOG/transpose_api_hits.txt"
        printf 'TRANSPOSE_API_HITS_END\n'
    fi
    printf 'archive=%s.tar.gz\n' "$RUN"
    printf 'C008_EVIDENCE_END\n'
    exit "$result"
}
trap finish EXIT

exec > >(tee "$LOG/run_full.log") 2>&1
: "${ASCEND_HOME_PATH:?source the CANN 9.1 set_env.sh first}"

{
    date -Is
    uname -a
    printf 'repo_head=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'repo_status_for_operator=\n'
    git -C "$REPO" status --short -- people/hongwei-2026/ConfusionTransposeD_2323
    printf 'ASCEND_HOME_PATH=%s\n' "$ASCEND_HOME_PATH"
    printf 'ASCEND_OPP_PATH=%s\n' "${ASCEND_OPP_PATH:-<unset>}"
    printf 'ASCEND_CUSTOM_OPP_PATH=%s\n' "${ASCEND_CUSTOM_OPP_PATH:-<unset>}"
    for path in "$ASCEND_HOME_PATH/version.cfg" "$ASCEND_HOME_PATH/version.info"; do
        if [ -f "$path" ]; then printf '%s\n' "--- $path"; sed -n '1,80p' "$path"; fi
    done
    npu-smi info || true
} > "$LOG/environment.log" 2>&1

python3 - "$ASCEND_HOME_PATH" "${ASCEND_OPP_PATH:-}" "$LOG" <<'PY'
from pathlib import Path
import sys

toolkit = Path(sys.argv[1])
opp_env = Path(sys.argv[2]) if sys.argv[2] else None
log = Path(sys.argv[3])

opp_roots = []
for root in (opp_env, toolkit / "opp", toolkit.parent / "opp"):
    if root and root.is_dir() and root not in opp_roots:
        opp_roots.append(root)

name_hits = []
for root in opp_roots:
    for path in root.rglob("*"):
        if path.is_file() and ("confusion" in path.name.lower() and "transpose" in path.name.lower()):
            name_hits.append(str(path))
(log / "tbe_file_hits.txt").write_text("\n".join(sorted(name_hits)) + ("\n" if name_hits else "<none>\n"))

symbol_hits = []
for root in opp_roots:
    for subdir in (root / "built-in", root / "vendors"):
        if not subdir.is_dir():
            continue
        for path in subdir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".json", ".ini", ".yaml", ".yml", ".xml"}:
                continue
            try:
                lines = path.read_text(errors="ignore").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                lower = line.lower()
                if "confusiontransposed" in lower or "npu_confusion_transpose" in lower:
                    symbol_hits.append(f"{path}:{number}:{line.strip()[:300]}")
(log / "tbe_symbol_hits.txt").write_text("\n".join(symbol_hits[:200]) + ("\n" if symbol_hits else "<none>\n"))

header_roots = [toolkit / "include", toolkit / "compiler" / "tikcpp"]
header_roots = [root for root in header_roots if root.is_dir()]
api_hits = []
for root in header_roots:
    for path in root.rglob("*.h"):
        name = path.name.lower()
        if "transpose" in name or "gather" in name:
            api_hits.append(str(path))
(log / "transpose_header_paths.txt").write_text("\n".join(sorted(api_hits)) + ("\n" if api_hits else "<none>\n"))

api_lines = []
for filename in api_hits:
    path = Path(filename)
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        continue
    for number, line in enumerate(lines, 1):
        if any(token in line.lower() for token in ("transpose(", "gather(", "transposemode", "gathermode")):
            api_lines.append(f"{path}:{number}:{line.strip()[:300]}")
(log / "transpose_api_hits.txt").write_text("\n".join(api_lines[:240]) + ("\n" if api_lines else "<none>\n"))

print(f"TBE filename hits: {len(name_hits)}")
print(f"TBE registration/source symbol hits: {len(symbol_hits)}")
print(f"Transpose/Gather header paths: {len(api_hits)}")
print(f"Transpose/Gather API declaration hits: {len(api_lines)}")
PY

cp "$0" "$LOG/"
