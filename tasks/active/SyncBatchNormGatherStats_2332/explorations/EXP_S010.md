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

Cloud system API comparison at 2a64a0b: system libopapi loaded, and global
vendor config lists p_relu_grad_update_custom, not this operator. Actual
device dispatch was not traced. Three int32 calls executed. On f16 input,
system vs custom mismatches were 0/2/0/4 for the four outputs; system vs
reference were 0/5/0/4. Native f32 differs from fp16-input results.
All three int64-count calls returned ACL 161002; the installed legacy
operator metadata and check_param list only int32 count. This does not
establish the custom operator or platform's int64 behavior.

The user supplied excerpts of installed legacy Python source: it sums
world-size counts, multiplies by reciprocal count, computes rsqrt with
vec_rsqrt_high_preci, and multiplies variance by unbias then momentum.
Dispatch to that specific implementation is not proven.

tests/s010_order_analysis.py models fp32 operation orders on the original
large input. Reciprocal plus factor-first reproduces 8/8 printed system
new_variance examples; division plus factor-first reproduces 7/8. Against
the local reference on the whole input, division plus momentum-first has
12 mismatches, division plus factor-first 14, reciprocal plus factor-first
18. The model is not device evidence for a modified custom kernel and
does not establish the platform golden. Keep the kernel unchanged pending
an experiment explicitly weighing both reference and system outputs.
