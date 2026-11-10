#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/tests/ut/build"
cmake -S "$ROOT/tests/ut" -B "$BUILD" -DGTEST_SOURCE_DIR="${GTEST_SOURCE_DIR:-}" -DSOC_VERSION="${ASCEND_COMPUTE_UNIT:-ascend910b}"
python3 "$ROOT/samples/test_reference.py"
cmake --build "$BUILD" --parallel "${CANNJUDGE_COMPILE_JOBS:-4}"
"$BUILD/op_host/p_relu_grad_update_op_host_ut"
python3 "$ROOT/samples/sample.py" generate --directory "$ROOT/samples/build-cpu"
export SAMPLE_DIR="$ROOT/samples/build-cpu"
"$BUILD/op_kernel/p_relu_grad_update_op_kernel_ut"
python3 "$ROOT/samples/sample.py" verify --directory "$SAMPLE_DIR"
if [ -x "$BUILD/op_api/p_relu_grad_update_op_api_ut" ]; then
    export ASCEND_CUSTOM_OPP_PATH="$ROOT/build/packages/vendors/p_relu_grad_update_custom${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    "$BUILD/op_api/p_relu_grad_update_op_api_ut"
fi
