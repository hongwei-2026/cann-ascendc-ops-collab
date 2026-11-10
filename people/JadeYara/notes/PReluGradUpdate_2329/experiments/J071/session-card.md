# J071 session: true-TBE FP32 split-C double buffering

2026-11-08. Seat JadeYara, sole implementer, performance phase.
One goal: migrate J059 input/output two-slot scheduling onto measured J070 actual-TBE arithmetic, only original FP32 split-C tileLen4096. A/C freeze final J070; B and standard use candidate. No arithmetic, Host, core count, tile length, General/Gather, phase retuning or queue merging.
Allowed: this experiment plus JadeYara personal ledger/dev-log. Production J035 and all previous experiments frozen. No official submission/commit/push. Cloud transfer and experiments authorized in original verified 910C.
Evidence: J059 11 inputs/66 rounds gain, 168 controls stable, actual three-engine overlap; J060 queue merge had unexplained controls and is excluded. J068-070 true TBE chain supersedes old where golden; preserve Inf/NaN and rounding behavior.
Limit: one candidate, 90 minutes active experiment work; no parameter sweep, no repeated run to force gates. Precision first: original 1063 groups standard+ABC; J059 lifecycle tails included in inherited corpus, actual TBE special inputs on split-C added. Full Task: inherited127 inputs, six balanced orders,350 calls/window, discard200/keep150, strict0.1us AND2% controls. General correctness retained and machine-code correspondence checked.
