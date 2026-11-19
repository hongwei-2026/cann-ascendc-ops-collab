# EXP_S008 - multi-tile output buffer synchronization

S007 summary reports C=524289 batch_mean has 249859 mismatches, mostly zero;
batch_invstd and new_variance also have 7 and 13 mismatches. The user's
terminal transcript is archived as EXP_S007/console-transcript.log; separate
original log files and device binaries are not yet available locally.

The kernel issues output DMA using VECCALC buffers and waits for MTE3_MTE2
before returning. The next tile can issue vector instructions that reuse the
same buffers; MTE3_MTE2 alone does not synchronize those vector writes.
S008 adds MTE3_V after the output DMAs and retains the existing MTE3_MTE2.
This is a candidate fix, not a proven cause. GM buffer length remains a
separate unverified hypothesis.

Cloud test at d75aa78: build and smoke PASS. Three identical large-case runs
each have batch_mean 0, batch_invstd 7, new_mean 0 and new_variance 13
mismatches out of 524289. Full strict matrix: 144 PASS, 1 FAIL, 0 ERROR,
12 PROBE. The batch_mean zeroing is resolved in these runs; other mismatches
remain. Terminal transcript: logs/EXP_S008/console-transcript.log.

The S007 matrix marks any mismatch FAIL. Using the numeric per-output
0.0001 mismatch-ratio rule from samples/sample.py, 7 and 13 are both below
the 52.4289-element limit for this constructed case. This is an extrapolation
of the sample rule and does not establish the hidden platform's acceptance.
See EXP_S009 for the remaining rounding analysis.
