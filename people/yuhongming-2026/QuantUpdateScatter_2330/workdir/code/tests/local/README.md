# Q001 local regressions

Run from any directory with Python, numpy and ml_dtypes installed:

    python -B test_q001.py

Use the full path to test_q001.py when running outside this directory.
The tests exercise reference semantics and a CPU model of tiling/copy/scatter.
They do not compile or execute the AscendC kernel and give no device precision or performance result.
CANN UT additionally has generated multicore, tail33, MODE_0/MODE_1 and inplace/optional-zp/reciprocal cases.

R001 extends the matrix to both index widths for each quantization dtype.
For native source-contract checks, run:

    python -B run_r001_contracts.py --cxx /path/to/g++

This compiles the actual registration, entry dispatch and diagnostic helper against
minimal API stubs. It does not compile the AscendC implementation or run NPU kernels.
