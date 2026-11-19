# EXP_S011 - isolate variance multiplication order

S010 established reproducible differences, not the platform's hidden golden.
The system fp16/int32 example matches 7/8 recorded variance positions in a
CPU model using division and factor-first multiplication, compared to 4/8
for the current model. On the full seeded C=524289 reference comparison,
the modeled mismatch count changes from 12 to 14. CPU results are not 910B
measurements.

This experiment copies the exact S008 kernel blob into an isolated project and
changes only the order of unbias and momentum multiplication. It retains the
current division, invstd, count handling, and all production source files.
The compact comparison verifies matching original inputs, identical reference
files, and exact unchanged outputs. System comparison uses the saved S010
fp16/int32 output, after verifying its quantized input bytes.

Local checks: candidate diff contains only the intended order change;
synthetic comparison validates all 12 cases and rejects an unexpected change
to batch_mean; missing inputs return ERROR. 910B build, compact, boundary and
full matrix results are pending. The selected 15 channels are not a random
sample, and the platform comparator and system device dispatch remain unknown.
