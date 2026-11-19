#!/usr/bin/env python3
"""Compare the installed custom kernel with reference on rounding boundaries."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "samples"))
import reference  # noqa: E402


def main():
    home = os.environ.get("ASCEND_HOME_PATH")
    if not home:
        raise RuntimeError("ASCEND_HOME_PATH is not set; initialize the CANN environment")
    executable = ROOT / "examples" / "build" / "sample"
    if not executable.is_file():
        raise RuntimeError("Run bash build.sh -j8 -e before this test")

    directory = ROOT / "samples" / "build-s006"
    for name in ("input", "output", "golden"):
        (directory / name).mkdir(parents=True, exist_ok=True)

    inputs = {
        "total_sum": np.array([[465.5, 442.5, 411.25, 465.0, 378.0, 100.0]], dtype=np.float32),
        "total_square_sum": np.array([[4080.0, 2088.0, 2590.0, 3672.0, 3806.0, 100.001]], dtype=np.float32),
        "sample_count": np.array([100], dtype=np.int32),
        "mean": np.array([3.0] * 6, dtype=np.float32),
        "variance": np.array([6.375, 4.45703125, 7.65234375, 3.19921875, 4.734375, 1.0], dtype=np.float32),
    }
    for name, value in inputs.items():
        value.tofile(directory / "input" / (name + ".bin"))
    names = ("batch_mean", "batch_invstd", "new_mean", "new_variance")
    expected = reference.impl(**inputs, momentum=0.1, eps=1e-5)
    for name, value in zip(names, expected):
        value.tofile(directory / "golden" / (name + ".bin"))

    vendor = ROOT / "build" / "packages" / "vendors" / "sync_batch_norm_gather_stats_custom"
    env = os.environ.copy()
    env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor) + os.pathsep + env.get("ASCEND_CUSTOM_OPP_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join((str(vendor / "op_api" / "lib"), str(Path(home) / "lib64"), env.get("LD_LIBRARY_PATH", "")))
    subprocess.run([str(executable), str(directory), env.get("DEVICE_ID", "0")], env=env, check=True)

    failed = False
    for name, golden in zip(names, expected):
        output_path = directory / "output" / (name + ".bin")
        actual = np.fromfile(output_path, dtype=golden.dtype)
        if actual.size != golden.size:
            raise RuntimeError(f"{name}: expected {golden.size} values, got {actual.size}")
        good = np.isclose(actual, golden, rtol=1e-4, atol=1e-4, equal_nan=True)
        print(f"{name}: mismatched={np.count_nonzero(~good)}/{golden.size}", flush=True)
        for index in np.flatnonzero(~good):
            print(f"  [{index}] actual={actual[index]!r} expected={golden[index]!r}", flush=True)
        failed |= not bool(np.all(good))
    print("FAILED" if failed else "PASSED", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
