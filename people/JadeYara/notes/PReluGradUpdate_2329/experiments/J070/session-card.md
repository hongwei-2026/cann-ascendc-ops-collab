# J070 session card

Member: JadeYara. Task: PReluGradUpdate_2329. Role: sole isolated implementer.
Goal: implement the proven compact-window broadcast fallback while preserving every J069 fastpath and the actual TBE numerical chain.
Read: root AGENTS, collaboration protocol, CLAIM, shared STATUS/SUMMARY/path-ledger, personal checkpoint/path-ledger, J068 contract/design, J069 source, J070-design proof/model/results.
Only write: people/JadeYara/notes/PReluGradUpdate_2329/experiments/J070/.
Parent owns shared/personal ledger updates and J069 device campaign/archive.
Do not edit J069, J070-design, production J035, other experiments, or existing dirty work. No commit, push, official submission, cloud upload/write/build/NPU.
Baseline: J069/candidate/code as on disk at initialization. Same-shape ND fastpaths remain unchanged when their exact contract holds. New route is isolated metadata and schMode 28/29/30.
Gate: local source/model validation only; CANN 9.3 ascend910_93 compilation, actual builtin-TBE comparison, physical layouts and full Task performance remain pending.

## Route registration
J070 / local implementation: continuous output windows bounded at P*, O(rank) exact input hulls, one bulk DMA plus bounded integer Gather per broadcast input, fixed cap 2048 and explicit UB accounting. Stop before cloud execution and report unresolved contracts.

TBE source directly computes output broadcast(g,x), then broadcasts mapped weights to it. Mapped weights do not enlarge that output. Unknown-rank graph compilation and mixed-format transformations are not verified.
