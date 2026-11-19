# EXP_S006 - fp32 normalization rounding

The baseline computes `sum * float32(1 / n)` and `square_sum * float32(1 / n)`.
The challenge reference computes fp32 divisions. In a CPU model, the two
forms yield different fp16-rounded `new_variance` values for some ordinary
inputs. This is a hypothesis for hidden WA, not a confirmed platform cause.

The S006 kernel uses `Div` with a vector of `n` for both normalizations.
It preserves the existing fp16 input/output rounding, reduction, count and
special-value logic. The remaining fp32 versus reference fp64 difference in
`new_variance` is not changed without device evidence.

`tests/s006_rounding.py` runs the custom six-channel sample executable with
fixed inputs and compares all four outputs to the shipped reference with the
sample tolerance. CPU models disagree on five `new_variance` values for these
inputs; only an actual 910B run can establish the kernel result. The test does
not run the built-in operator or reproduce the hidden tests.

Local checks: Python AST parse and `git diff --check` passed. 910B build,
sample, boundary test, and platform submission remain pending.
