# J028: fixed-tile low-precision split-C input prefetch

Scope: JadeYara/PReluGradUpdate_2329 only. User authorized this bounded performance experiment after J027. Preserve J026/J027 and all previous evidence; no push/submission/network changes.
J028 derives from J026, not J027, to isolate prefetch. Only low-precision input queues get two slots. Keep tile/core/routing unchanged; explicitly budget 39 B/element against 90% of reserved UB. Prefetch next batch before current Compute, retain single weights/output slots. FP32 keeps its serial schedule.
Gate: first build J028IO and compare with J027IO identity diagnostic, four counterbalanced ABBA/BAAB rounds, 3 existing large shapes plus 1 new adjacent shape, all dtypes. Stop if no consistent low-precision gain beyond control/state variability; no parameter grid. Only if gate passes, build real J028, run 708 regression plus new shapes and alternating A/B. Estimated 20–30 minutes excluding transfer.

Diagnostic gate FAILED: no stable benefit. See [RESULTS.md](RESULTS.md). Real J028 was NOT device compiled/tested.
