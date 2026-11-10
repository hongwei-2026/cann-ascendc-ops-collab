# J060: paired-lifecycle full pipeline
PReluGradUpdate_2329 / JadeYara. User authorized continuing the effective full pipeline, 2026-11-07 CST.
One independent candidate: paired g/x input and dx/update output queues for fixed4096 FP32 split-C, preserving next-input prefetch, residency, original math and Host/core policy.
A/C freeze J059; B is the paired-lifecycle implementation. Never overwrite J035/J056/J059. Scope local/remote J060 and this task ledgers.
Hypothesis: reduce queue/flag/scalar management and startup costs seen in J059 while retaining three stages. Not a tile/core sweep; old J029 was not this full-FP32-double-slot pipeline.
Input slot stride33280B with x at+16640B and extra256B after input queue. UB186112B, available188416B, headroom2304B. Both slots preserve binary operand phases.
Necessary correctness and one same-process complete-Task campaign. Stop this candidate if tiny or slower. No official submission, commit, push, or new persistent system settings.
