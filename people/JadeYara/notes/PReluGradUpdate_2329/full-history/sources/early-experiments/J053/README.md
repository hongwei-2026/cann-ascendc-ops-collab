# J053: same-process, fixed-allocation A/A and A/B/C

User authorized continuing the J050-J052 timing diagnosis. One goal: compare the CANN9.3 rebuilt J035 baseline with independent-buffer per-tile conversion and cached conversion while holding process, tensors and stream fixed. No platform operation, submission, push, network/settings changes or installation of new software.

Test-only aliases PReluGradUpdateA/B/C in one private package derive mechanically from J050/J051/J052. Only registration identifiers and file names change. Require compiled function-body SHA correspondence to original variants before interpreting timings; preserve originals. This avoids guessing the existing binary's hidden direct-launch ABI.

Correctness: all780 finite three-dtype inputs and5184 per-invocation stress checks for each alias. Same-process A/A labels invoke the same A function. Ordinary profiling keeps prior timing boundary: each kernel is synchronized;350calls/window, discard first200, median last150 Task Duration and AIV metrics. All five device addresses logged; no allocation/free between variants of a case. Predeclare balanced variant orders, randomized case order, low/high reuse, tail and small controls. Retain per-round deltas and A/A spread. Pipeline active times overlap and are not exclusive latency. A/A or unchanged controls outside0.1us/2% limits require explaining the drift before accepting a dispatch. If a generic reuse rule is supported, validate held-out shapes and original-path precision before preparing a user-operated replacement and rollback. No official-test-number specialization.

Writable scope: people/JadeYara/PReluGradUpdate_2329/experiments/J053 and this task's experiment records; cloud /home/developer/2329-JadeYara-20261104-1812/J053*.


本地测试入口已补于 [control/README.md](control/README.md)，含运行步骤及实际验证范围。当前仍待 CANN 编译和 NPU 运行，J035 基线未替换。
