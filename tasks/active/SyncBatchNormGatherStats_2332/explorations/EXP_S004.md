# EXP_S004 - preserve non-finite values in the common fp16 round trip

Date: 2026-11-17
Owner: yuhongming-2026
Scope: precision only; no performance work

## Trigger

S003 was submitted to the platform and remained Wrong Answer. The platform
shows 50 tests total: tests 1-5 are public and pass; tests 6-50 are hidden.

## Hypothesis

`QuantFp16Vec` can erase NaN or +/-Inf before values reach the declared-float
`new_variance` output. This is observable for `count==1`, where the unbiased
factor is +Inf and `Inf * 0` may be NaN. S003 only repaired the final half
store path used by `batch_invstd`.

## Change

In `op_kernel/sync_batch_norm_gather_stats.h`, restore non-finite source
values in the temporary half tensor after the vector cast and before the
float round-trip. A `V_S` event synchronizes the scalar check. Finite values
continue through the existing vector path unchanged.

## Verification

- `git diff --check`: passed.
- Python reference probes: confirmed `count==1` produces NaN or +/-Inf;
  no finite clamp was added.
- `bash build.sh -j8 -e`: **not runnable on this Windows host**;
  WSL reports `/bin/bash` is unavailable. Full output is archived in
  `explorations/logs/EXP_S004/build_e.log`.
- Platform re-submission: not performed in this session; S003 platform result
  remains WA until a new package is uploaded.

## Conclusion

S004 is a minimal precision fix, but it is not platform-validated yet.
