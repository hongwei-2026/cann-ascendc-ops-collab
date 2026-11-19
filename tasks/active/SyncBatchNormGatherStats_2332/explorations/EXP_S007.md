# EXP_S007 - broad precision matrix

The original ZIP contains one samples/case.json, not the hidden 50-case list.
The existing sample and kernel UT both use C=6, world=1, float32/int32.
S006 passed that sample and its rounding fixture on 910B; this does not prove
that the platform passed.

S007 adds a configurable ACLNN runner and deterministic 157-case matrix.
Single-world, single-count cases compare all four outputs with the existing
reference at the sample tolerance. They exercise four dtype combinations,
rank 1 and 2, aligned and unaligned channel counts, large channels, count
values, attributes, rounding boundaries and non-finite results.

Multi-world and multi-count cases are PROBE only: first/sum alternatives are
printed separately and must be adjudicated against an authoritative source.
All inputs and outputs are saved for reproduction. Exact special-value
comparison is a diagnostic and may differ from platform grading rules.

Local validation: generator self-test only. No 910B run or platform result
for S007 is claimed. See CLOUD-S007.md for execution and log collection.
