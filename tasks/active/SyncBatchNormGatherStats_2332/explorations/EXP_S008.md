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

The test matrix now accepts --case and preserves the original matrix index
and random seed. CLOUD-S008.md repeats the original large case in separate
directories, then runs the full matrix. Device results are pending.
