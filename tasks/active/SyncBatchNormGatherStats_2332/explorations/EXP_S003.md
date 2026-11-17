# EXP_S003 - platform WA precision probe

Date: 2026-11-17
Owner: yuhongming-2026
Scope: precision only; no performance work

## Hypothesis

The fp16 output path can lose a non-finite `batch_invstd` value twice: the
kernel repairs negative or zero `batch_var + eps` after vector math, then
`StoreAsInputDtype<half>` applies another `CAST_RINT`. The reference preserves
NaN and positive infinity.

## Evidence

- `ComputeTile` repairs NaN/+inf before output.
- The fp16 store path performs a second vector cast after that repair.
- The official task description says `sampleCount` is per-device, so the
  existing count summation and world reduction remain unchanged.
- The local reference regression with negative `eps` returns `batch_invstd=NaN`.

## Change

After the final vector cast, re-apply source NaN and +/-inf values through the
scalar half conversion before DMA. Finite values retain the existing vector
path.

## Verification

- `git diff --check`: passed.
- Reference special-value probe: passed (`batch_invstd` is NaN).
- Cloud/platform rebuild and resubmission: pending; no PASS is claimed.

## Next step

Build on the 910B environment and rerun the platform hidden points. If WA
remains, use the resulting failing case to choose the next single ledger path.
