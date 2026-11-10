# J029: paired input/output queues in split-C

User authorized a bounded investigation of per-batch transfer/queue overhead. Scope: JadeYara/PReluGradUpdate_2329 only; preserve J026/J027/J028, no submission/push/network change.
Evidence: J027/J028 IO diagnostics retain about 4us Scalar; each split-C batch has four independent queue lifecycles (16 Alloc/EnQue/DeQue/Free calls), while both inputs must be ready before Compute and both outputs are produced together. Packing grads/features into one queue tensor and dx/update into another halves queue operations without changing four GM copies, mathematical operations, tile, core count or UB bytes. Non-overlapping regions use aligned tile offset; one queue event orders both copies. This is a synchronization/queue candidate, not a reduction in GM traffic.
Two-batch full-buffer fusion would require 35*2*4096=286720 B at the target low-precision tile, above 188416 B usable UB, so do not shrink tile or start a fusion scan.
One real candidate from J026. Compile, 708 existing precision cases plus new adjacent split-C boundaries, then four ABBA/BAAB rounds on a bounded set of large shapes and generic control. Stop if no stable benefit; no second candidate this turn.

723 precision PASS; no stable end-to-end gain in four counterbalanced rounds. NOT promoted. See [RESULTS.md](RESULTS.md).
