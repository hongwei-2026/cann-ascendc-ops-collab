# MultilabelMarginLoss 2327: M009 candidate

Based on hongwei M006 (platform cases 1-5 passed). M009 has not yet been built
with CANN or run on an NPU. Keep the original M006 ZIP for rollback.

The kernel loss/count algorithm is unchanged. M009 removes the fixed 16MB
workspace reservation: none and single-core reductions use zero workspace;
multi-core mean/sum use only 32-byte-aligned FP32 losses[N]. Small reductions
(N<=8, C<=32, K<=32, N*C*K<=4096) finish on one core in row order.
The kernel uses the raw workspace exclusively for its row losses; it has no
Matmul/HCCL/software-sync system-workspace consumers. Multi-core hardware
SyncAll and the core0 ordered final sum remain in place.

The existing example still defaults to the original seed42 [4,6] mean case.
Optional arguments now support broader precision/scheduling checks. Host and
Kernel UT fixtures were updated for the actual tiling structure.

## Cloud check

Extract this ZIP into a separate 2327 directory, enter its code directory,
and load the same CANN environment used for M006. Then run:

```bash
DEVICE_ID=0 bash tests/run_m009_cloud.sh
```

This verifies source fingerprints, builds all three keys, runs the original
sample, and checks 72 cases covering all dtypes/reductions, K!=C, duplicates,
negative truncation, empty/full target sets, small/large/tiled rows, and the
single/multiple-core paths. Logs, inputs, golden and actual outputs are packed
under logs/EXP_M009_*.tar.gz even if a check fails. A missing build-copy path
is reported; inspect the build tool's source-copy directory in that case.

For only the original sample:

```bash
DEVICE_ID=0 bash build.sh -j4 -e
```

After correctness passes, compare platform times against M006 under the same
conditions. No speedup or hidden-case PASS has been measured locally.
