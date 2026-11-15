# QuantUpdateScatter_2330: Q001 local candidate

Source: workdir/code/. Restored from the original local project and the supplied handoff.
Q001: aligned parallel copy, bounded indices cache, parameter reuse per column tile.
Reference: fp32 arithmetic and direct reciprocal multiplication. Kernel UT: MODE_0 fixture.
No cloud run, push, CANN compilation, NPU accuracy or performance claim.
Records: tasks/active/QuantUpdateScatter_2330/.

## R001 runtime fix

Input registration and kernel dispatch now cover both int32 and int64 indices
for each supported quantization triple. The original template registered only
fp16+int32 and bf16+int64 indices. Both missing combinations are now included.
The example also reports the exact failing ACL expression and aclGetRecentErrMsg.

9 CPU regressions and 1152 schedule comparisons pass. Native source-contract
checks compile the actual registration/entry/error helper with minimal API stubs.
This revised candidate has not been compiled with CANN or retested on the platform.
See tasks/active/QuantUpdateScatter_2330/explorations/EXP_R001-runtime.md.
Rebuild and submit the complete workdir/code tree so registration and kernel match.

## Current candidate: Q003 resource allocation

Q002 received user-reported platform Pass (screenshot time 14:20:36).
Its complete local source is preserved in explorations/Q003/baseline/.
The current workdir/code candidate removes unused parameter buffers and sizes
raw buffers and per-core index reservation by actual types and input presence.
Computation, copy, queue ownership and SyncAll definitions are unchanged from Q002.

Local checks: 2432 Q003/Q002/scalar comparisons across four UB capacities,
13824 actual-host and 12096 independent resource plans, 10 regression tests,
1152 schedule comparisons, and the earlier source-contract checks pass.
These use CPU API stubs; CANN compilation and the full 50-case platform retest
are still required. No Q003 latency improvement is claimed before that retest.
Report: tasks/active/QuantUpdateScatter_2330/explorations/EXP_Q003-ub.md.
