# EXP_S009 - remaining precision and tolerance

S008 910B data at d75aa78: three repeated large cases have mismatch counts
0 / 7 / 0 / 13 for batch_mean / batch_invstd / new_mean / new_variance.
The matrix's strict rule fails an output for any mismatch. The sample
`samples/sample.py` instead permits bad <= 0.0001 * output_size for each
output. Extrapolating its thresholds gives 52.4289 allowed at size 524289,
so 7 and 13 fall below that sample-derived limit. The platform's hidden-case
tolerances and comparator have not been established.

`tests/s009_rounding_analysis.py` reconstructs the original seeded input.
The CPU fp32 model differs from reference on 12 new_variance elements; its
values match the eight mismatch examples in the cloud log. The CPU model
does not reproduce the 7 invstd differences. These observations do not
establish independent built-in device semantics.

S007 reports both strict status and the separately labeled sample-rule status.
The strict status and process exit code are unchanged. Do not change the
kernel arithmetic merely to fit a modeled reference without device evidence.
