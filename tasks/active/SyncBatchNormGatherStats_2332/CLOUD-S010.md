# S010 cloud reproduction

Use the collaboration repository on the 910B machine after publishing S010.
From the repository root:

```bash
git pull --ff-only origin main
export DEVICE_ID=0
export S010_LOG="$PWD/tasks/active/SyncBatchNormGatherStats_2332/explorations/logs/EXP_S010"
mkdir -p "$S010_LOG"
cd people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code
set -e -o pipefail
python3 -B tests/s010_compact.py --self-test 2>&1 | tee "$S010_LOG/selftest.log"
bash build.sh -j8 -e 2>&1 | tee "$S010_LOG/build_e.log"
set +e
python3 -B tests/s010_compact.py --repeat 3 --output samples/build/s010 2>&1 | tee "$S010_LOG/compact.log"
compact_status=${PIPESTATUS[0]}
set -e
echo "compact exit: $compact_status"
cp samples/build/s010/results.json "$S010_LOG/results.json"
```

The compact command returns 1 on any strict mismatch or execution error;
this is expected for a reproduction and does not by itself classify the root
cause. Save all logs, including failed runs. Do not add samples/build to Git.
The result compares to the local reference, not to an independent built-in.

## Read-only built-in discovery

Run on the same cloud host; this searches the installed CANN environment and
does not execute any operator. Share the resulting built-in-discovery.log.

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
export S010_LOG="$PWD/tasks/active/SyncBatchNormGatherStats_2332/explorations/logs/EXP_S010"
mkdir -p "$S010_LOG"
{
  echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset}"
  echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-unset}"
  echo "Matching installed paths:"
  find -L "${ASCEND_HOME_PATH:?CANN environment is not initialized}" -iname '*sync*batch*norm*' -print 2>&1
  echo "Matching dynamic symbols in installed operator API libraries:"
  find -L "$ASCEND_HOME_PATH" -type f \( -name '*opapi*.so*' -o -name '*nnopbase*.so*' \) -print |
    while IFS= read -r lib; do
      nm -D --defined-only "$lib" 2>/dev/null | grep -iE 'sync.*batch.*norm' | sed "s|^|$lib: |" || true
    done
} 2>&1 | tee "$S010_LOG/built-in-discovery.log"
```

Matching names only identify candidates: they do not prove an independent
operator ran, or that an API has the same signature and semantics.

## System ACLNN comparison

The installed header describes an API with two outputs and in-place mean and
variance. This candidate test links only the system ACLNN library, removes
custom OPP and preload variables from the runner process, prints the actual
library, and records possible globally installed vendor registrations. These
checks do not alone prove which device kernel was dispatched.

From the repository root after publishing the test:

```bash
git pull --ff-only origin main
export DEVICE_ID=0
export S010_LOG="$PWD/tasks/active/SyncBatchNormGatherStats_2332/explorations/logs/EXP_S010"
mkdir -p "$S010_LOG"
cd people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code
set -e -o pipefail
python3 -B tests/s010_builtin_compare.py --self-test 2>&1 | tee "$S010_LOG/builtin-selftest.log"
mkdir -p samples/build
g++ -std=c++17 -Iexamples -I"$ASCEND_HOME_PATH/include" \
  tests/s010_builtin_runner.cpp "$ASCEND_HOME_PATH/lib64/libopapi.so" \
  -L"$ASCEND_HOME_PATH/lib64" -lascendcl -lnnopbase -ldl \
  -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" -o samples/build/s010_builtin_runner \
  2>&1 | tee "$S010_LOG/builtin-build.log"
{
  echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-unset}"
  if [ -d "${ASCEND_OPP_PATH:-/nonexistent}/vendors" ]; then
    find "$ASCEND_OPP_PATH/vendors" -maxdepth 2 -type f -print
    if [ -f "$ASCEND_OPP_PATH/vendors/config.ini" ]; then
      cat "$ASCEND_OPP_PATH/vendors/config.ini"
    fi
  else
    echo "No global vendors directory found at ASCEND_OPP_PATH."
  fi
} 2>&1 | tee "$S010_LOG/builtin-vendors.log"
set +e
python3 -B tests/s010_builtin_compare.py 2>&1 | tee "$S010_LOG/builtin-compare.log"
compare_status=${PIPESTATUS[0]}
set -e
echo "comparison exit: $compare_status"
cp samples/build/s010-builtin/results.json "$S010_LOG/builtin-results.json"
```

`native` passes the original dtype; for original f32 cases `fp16_input`
passes values converted to fp16, then compares the resulting values after
conversion back to float32. These are distinct API-level experiments, not a
claim that either reproduces the platform's graph or judge. Unsupported dtype
or launch failures are reported as ERROR. The custom comparison is skipped if
the earlier S010 input files are absent or differ byte-for-byte.
