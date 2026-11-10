# J077 — True-TBE contract and reproducible build/UT repair

Date: 2026-11-08 (Asia/Shanghai). Owner: JadeYara. One implementer. User explicitly authorized correcting the TBE contract, single-variable optimization from official1221802, and closing build/UT reproduction gaps.

This first bounded work item repairs documentation, build target selection, and obsolete UT only. Baseline: official submission1221802, SHA256482fb0fa80e13d964fed1e02d98b55b8538c6c29cb4c5b28e8d7d952c20e79ea. Runtime op_host/op_kernel remain frozen. Performance changes belong to a subsequent independent experiment from the same official baseline.

Writable paths: E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/notes/PReluGradUpdate_2329; this task shared analysis/SUMMARY.md, path-ledger.md, dev-log.md. C:/Users/lin/Documents/Codex/2026-11-05/task-2/publish-repo is the read-only historical source. No writes to other members, historical experiments or production J035. No official submission or public push.

Validation: actual Host tiling/shape tests against SDK where available; kernel CPU simulator when supported; true-TBE/ACL regression on the verified existing910C; original910b and explicit910_93 runs reported separately. Do not replace a failing test with SKIP or infer device success from local models. Bound diagnostic work to90 active minutes; unresolved cases preserve logs and stop with an explicit DEC.
