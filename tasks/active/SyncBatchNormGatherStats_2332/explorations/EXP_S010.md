# EXP_S010 - compact recorded precision differences

S008 logged seven invstd and thirteen new_variance differences out of
524289 channels; only eight new_variance indices were printed. This test
extracts those fifteen recorded channels from the original seeded input.
It repeats the compact case for four input/count dtype combinations.

Local self-test PASS: each compact reference output matches the corresponding
original reference value; no device was run. The local reference is not an
independent built-in oracle. A compact failure demonstrates a difference from
that reference; it does not prove a platform hidden case has the same inputs
or determine which result matches the built-in device operator.

The S007 probes compared the custom kernel to candidate formulas, not to an
independent built-in implementation. Their agreement establishes the custom
kernel's observed behavior, not the platform's semantic contract.

Next: run CLOUD-S010.md on 910B, archive the complete log and results.json,
and compare the cases with an independently callable built-in if available.
Do not alter arithmetic based only on this test.
