# J057 session card

- Task: PReluGradUpdate_2329, seat JadeYara, performance stage.
- Goal: Test whether one compact full-channel FP32 row per core improves large-channel latency relative to frozen J035 and the already-tested J054 two-row split-C layout.
- Writes: This personal J057 directory and personal path-ledger/dev-log only; matching remote J057 directory under /home/developer/2329-JadeYara-20261104-1812 after connection and job checks.
- Controls: A = J035, B = compact full-row, C = unchanged J054 compact two-row; same process/tensors/stream and balanced order. The reused C is a mechanistic control, not a renewed J054 tuning campaign.
- Scope: Only existing FP32 split-C, C divisible by 64, old tile <=4096, full C <=8192. Keep Host, other dtypes, small inputs and all existing paths unchanged. No official submission or public push.
- Stop: One bounded campaign, at most 90 minutes after device access. Any correctness failure blocks performance. Require six-round direction agreement and >2% target benefit; keep the existing strict A/A and fallback controls. Do not repeat to force a passing gate.
- Baselines: J035 immutable. J056 official submission 1191764 Pass confirmed by user: 7.21/7.16/7.19/5.58/5.84 us. Official shape/dtype unknown.
- Existing notes: J054 compact two-row was slower than J035 in all six rounds. This experiment changes ownership to whole contiguous channel rows and eliminates the inner per-row Mul/DMA gap; it does not just rerun the same kernel.
- Current blocker: AtomGit bootstrap times out waiting for refreshed connect_url. No remote mutation or device run has occurred for J057. Resume only after checking remote existing jobs.
- Stop That Shit skill 0.2.5 read and applied in change scope; no claim that Guard hooks are armed.
