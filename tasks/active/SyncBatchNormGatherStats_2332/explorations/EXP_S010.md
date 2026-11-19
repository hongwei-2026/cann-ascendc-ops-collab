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

Cloud at 1c1c29b: self-test and build/example PASS (four outputs 0/6).
All four dtype combinations, each repeated three times, produced the same
reference mismatch counts: batch_mean 0/15, batch_invstd 7/15, new_mean 0/15,
new_variance 8/15. The values agree with the corresponding S008 printed
examples. Strict and sample-derived comparison statuses both fail. The
fifteen channels were selected because they differ; their mismatch rate is
not representative of a random case or a platform hidden case.

The user-provided complete terminal transcript is archived in
logs/EXP_S010/console-transcript.log. The individual original cloud log files
and results.json remain on the cloud host. Next, use CLOUD-S010.md to discover
built-in interfaces, then compare independently if the environment supports it.
Do not alter arithmetic based only on this test.

Discovery found the system CANN ACLNN symbols and 910B legacy kernel files.
The user supplied the system header: its API takes float momentum and epsilon,
returns batch mean and inverse standard deviation, and updates mean and variance
in place. The discovered entry point and installed device files are candidates,
not proof that a specific device kernel was dispatched. See CLOUD-S010 for
the isolated comparison and the native/fp16-input distinction.
