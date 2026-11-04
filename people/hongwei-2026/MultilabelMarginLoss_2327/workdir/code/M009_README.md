# MultilabelMarginLoss 2327: M009 candidate

Based on hongwei M006. Cloud EXP_M009: none/tiny PASS; multi-core mean/sum
hit ACL 507035 without the system GM prefix. **M010** restores 16MB system
workspace + `GetUserWorkspace` for multi-core only.

Still: none and single-core (incl. tiny mean/sum) use workspace=0; multi-core
mean/sum use `WS_SYS_SIZE + aligned losses[N]`. Loss/count algorithm unchanged.

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
