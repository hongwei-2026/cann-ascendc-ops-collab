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
