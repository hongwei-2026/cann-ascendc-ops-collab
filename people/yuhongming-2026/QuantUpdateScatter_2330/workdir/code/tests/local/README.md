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

For Q003 allocation and actual-class comparisons against the saved passing Q002:

    python -B run_q003_resource_model.py --cxx /path/to/g++

This compiles the actual resource-plan block extracted from the host source and
actual kernel classes with the synchronous CPU API model. It does not validate
CANN compiler behavior, asynchronous device ordering or NPU latency.

For Q005 diagnostic-runner plumbing checks:

    python -B run_q005_support.py --cxx /path/to/g++

See ../../tools/README_Q005.md. These checks compile the diagnostic executable against synchronous CPU ACL stubs, not CANN. They do not establish device precision, asynchronous scheduling or speedup.
