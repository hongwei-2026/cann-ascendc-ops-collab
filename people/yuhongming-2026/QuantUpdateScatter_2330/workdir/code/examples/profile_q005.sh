#!/bin/bash
# Build the diagnostic executable only. Build/check the operator with -e first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${ASCEND_HOME_PATH:?source the CANN set_env.sh first}"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash examples/profile_q005.sh NEW_OUTPUT_DIRECTORY [--case-id ID ...]"
  exit 2
fi
OUT="$1"
shift
if [[ -e "$OUT" ]]; then
  echo "Choose a new output directory; existing evidence will not be overwritten: $OUT"
  exit 2
fi
export ASCEND_CUSTOM_OPP_PATH="$ROOT/build/packages/vendors/quant_update_scatter_custom${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$ROOT/build/packages/vendors/quant_update_scatter_custom/op_api/lib:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
cmake -S "$ROOT/examples" -B "$ROOT/examples/build" 2>&1 | tee "$OUT/configure.log"
cmake --build "$ROOT/examples/build" --target profile_case --parallel "${CANNJUDGE_COMPILE_JOBS:-4}" 2>&1 | tee "$OUT/runner-build.log"
python3 "$ROOT/tools/profile_q005.py" generate --directory "$OUT/cases"
python3 "$ROOT/tools/profile_q005.py" run --directory "$OUT/cases" --output "$OUT/evidence" \
  --runner "$ROOT/examples/build/profile_case" --device "${DEVICE_ID:-0}" "$@"
