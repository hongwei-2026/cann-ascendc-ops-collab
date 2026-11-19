# S011: isolated variance-order experiment on 910B

S011 tests one candidate only: compute `(batch_var * unbias) * momentum`
instead of `batch_var * (unbias * momentum)`. The original S008 project
and submission sources are not modified.

Prerequisites: use the same cloud checkout that ran S010, retaining
`samples/build/s010` and `samples/build/s010-builtin`. The comparison
validates exact input and reference files before comparing. The system
outputs were obtained with the system library; actual device dispatch was
not traced. int64 system cases are explicitly skipped.

Run from the repository root:

```bash
git pull --ff-only origin main
export DEVICE_ID=0
export S011_CODE="$PWD/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code"
export S011_RUN="$(date +%Y%m%d-%H%M%S)-$$"
export S011_LOG="$PWD/tasks/active/SyncBatchNormGatherStats_2332/explorations/logs/EXP_S011/$S011_RUN"
export S011_PROJECT="$S011_CODE/samples/build/s011-$S011_RUN"
mkdir -p "$S011_LOG"
(
  set -e -o pipefail
  cd "$S011_CODE"
  python3 -B tests/s011_prepare.py "$S011_PROJECT"
  git diff --no-index -- op_kernel/sync_batch_norm_gather_stats.h "$S011_PROJECT/op_kernel/sync_batch_norm_gather_stats.h" || test "$?" -eq 1
  cd "$S011_PROJECT"
  python3 -B tests/s010_compact.py --self-test
  bash build.sh -j8 -e 2>&1 | tee "$S011_LOG/build_e.log"

  set +e
  python3 -B tests/s010_compact.py --repeat 3 --output samples/build/s011-compact 2>&1 | tee "$S011_LOG/compact.log"
  compact_status=${PIPESTATUS[0]}
  python3 -B tests/s006_rounding.py 2>&1 | tee "$S011_LOG/rounding.log"
  rounding_status=${PIPESTATUS[0]}
  python3 -B tests/s007_matrix.py --output samples/build/s011-matrix 2>&1 | tee "$S011_LOG/matrix.log"
  matrix_status=${PIPESTATUS[0]}
  python3 -B "$S011_CODE/tests/s011_compare.py" \
    --baseline "$S011_CODE/samples/build/s010" \
    --candidate "$S011_PROJECT/samples/build/s011-compact" \
    --system "$S011_CODE/samples/build/s010-builtin" \
    --output "$S011_LOG/comparison.json" 2>&1 | tee "$S011_LOG/comparison.log"
  comparison_status=${PIPESTATUS[0]}
  set -e

  printf 'compact=%s rounding=%s matrix=%s comparison=%s\n' \
    "$compact_status" "$rounding_status" "$matrix_status" "$comparison_status" | tee "$S011_LOG/status.log"
  [ ! -f samples/build/s011-compact/results.json ] || cp samples/build/s011-compact/results.json "$S011_LOG/compact-results.json"
  [ ! -f samples/build/s011-matrix/results.json ] || cp samples/build/s011-matrix/results.json "$S011_LOG/matrix-results.json"
  test "$comparison_status" -eq 0
) 2>&1 | tee "$S011_LOG/console.log"
```

The compact and matrix scripts exit nonzero on any strict mismatch; their
status is evidence, not necessarily an infrastructure error. A nonzero
comparison status means missing/mismatched inputs or unexpected changes to
other outputs; do not treat it as a valid comparison.

Archive every file in this run's EXP_S011 log directory, including failures,
and share the complete console/status/comparison/matrix outputs. Do not add
the generated project, binaries or `samples/build` to Git. No candidate is
approved for platform submission until the 910B results are reviewed.
