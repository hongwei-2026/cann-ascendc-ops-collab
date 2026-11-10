# J067: native FP16 vector data path
PReluGradUpdate_2329 / JadeYara / performance stage / 2026-11-07.
Goal: independently test native half multiply/Compare/Select after the actual P031 unaligned Duplicate fix, with J035 A/C and unchanged Host/allocation/TPipe ownership.
Rationale: P030 FP32 promotion did not fix platform case4; P031 alignment did. Therefore native half itself was not independently disproven. User now explicitly states logical comparison thresholds float1e-4/half1e-3 and generalization. Keep stricter existing exact logical checker first; never loosen checks to force a pass.
Only J067 and personal notes/ledger/dev-log change. Keep all original FP32/BF16 paths, queue allocation/routing/keys/tile/cores, weight preparation, update not reduction. No J065/ShortS/J056/J059 combination.
Read native FP16 products and comparison semantics in actual device tests including NaN/Inf/zero/subnormal boundaries. One campaign, no parameter scan, no official submission/commit/push. Preserve failures and stop on correctness error.
