#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/mnt/workspace/cann-ascendc-ops-collab}"
BUILD_CODE="${2:-/mnt/workspace/prelu2329/code}"
SOURCE_CODE="${REPO_ROOT}/people/hongwei-2026/PReluGradUpdate_2329/workdir/code"
KERNEL_REL="op_kernel/p_relu_grad_update.h"
KEY_REL="op_kernel/p_relu_grad_update_tiling_key.h"

if [[ ! -d "${SOURCE_CODE}" || ! -d "${BUILD_CODE}" ]]; then
  echo "ERROR: source or build code directory is missing" >&2
  exit 2
fi

COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "repo_commit=${COMMIT}"
echo "source_kernel=$(sha256sum "${SOURCE_CODE}/${KERNEL_REL}" | awk '{print $1}')"
echo "build_kernel=$(sha256sum "${BUILD_CODE}/${KERNEL_REL}" | awk '{print $1}')"

if ! diff -q "${SOURCE_CODE}/${KERNEL_REL}" "${BUILD_CODE}/${KERNEL_REL}" >/dev/null; then
  echo "ERROR: CodeArts build tree is stale; copy workdir/code again before building" >&2
  exit 3
fi

if grep -R -n -F 'SyncFunc<HardEvent::S_V>' "${BUILD_CODE}/op_kernel"; then
  echo "ERROR: stale SyncFunc implementation is still present" >&2
  exit 4
fi

if ! grep -q -F 'SetFlag<HardEvent::S_V>' "${BUILD_CODE}/${KERNEL_REL}"; then
  echo "ERROR: expected CANN 9.1 SetFlag/WaitFlag fix is missing" >&2
  exit 5
fi

# P029: 三 dtype；禁止 Cast / ToFloat / DtypeBridge 显式特化
if ! grep -q '#define PRELU_ENABLE_FP16 1' "${BUILD_CODE}/${KEY_REL}" || \
   ! grep -q '#define PRELU_ENABLE_BF16 1' "${BUILD_CODE}/${KEY_REL}"; then
  echo "ERROR: P029 expects PRELU_ENABLE_FP16/BF16 default 1" >&2
  exit 6
fi
if grep -nE '\bCast\s*\(|ToFloat\s*\(|struct DtypeBridge' "${BUILD_CODE}/${KERNEL_REL}" | grep -v '//' >/dev/null; then
  echo "ERROR: Cast/ToFloat/DtypeBridge must not appear in kernel (contaminates all keys)" >&2
  exit 7
fi
if ! grep -q 'P029' "${BUILD_CODE}/${KERNEL_REL}"; then
  echo "ERROR: expected P029 native-dtype kernel marker" >&2
  exit 8
fi

echo "CodeArts source verification: PASS"
