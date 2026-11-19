#!/usr/bin/env python3
"""Replay recorded S008 differences in a compact, non-platform test."""
import argparse
import json
import os
from pathlib import Path
from unittest import mock
import numpy as np
import s007_matrix as matrix

INDICES = np.array([69528, 84745, 354501, 366094, 393032, 498326, 503890,
                    188307, 201472, 215481, 301966, 368953, 377353, 384637, 454808])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=matrix.ROOT / "samples" / "build" / "s010")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    selected = matrix.select_cases(matrix.cases(), "large_f32_i32_c524289")
    if len(selected) != 1:
        raise ValueError("Original large case missing or ambiguous")
    index, original = selected[0]
    source = matrix.inputs_for(original, 20261119 + index)
    full_golden = matrix.golden(original, source)
    assert len(INDICES) == len(set(INDICES)) == 15
    cases = []
    for dtype in ("f32", "f16"):
        for count_type in ("i32", "i64"):
            ft = np.float16 if dtype == "f16" else np.float32
            ct = np.int64 if count_type == "i64" else np.int32
            case = dict(original, name=f"compact_{dtype}_{count_type}", dtype=dtype,
                        count_type=count_type, c=len(INDICES))
            data = dict(total_sum=source["total_sum"][:, INDICES].astype(ft),
                        total_square_sum=source["total_square_sum"][:, INDICES].astype(ft),
                        sample_count=source["sample_count"].astype(ct),
                        mean=source["mean"][INDICES].astype(ft),
                        variance=source["variance"][INDICES].astype(ft))
            for compact, full in zip(matrix.golden(case, data), full_golden):
                assert np.array_equal(compact.astype(np.float32), full[INDICES].astype(np.float32),
                                      equal_nan=True)
            cases.append((case, data))
    if args.self_test:
        print("SELF-TEST PASS: 15 original channels, four dtype combinations; no device executed")
        return 0
    home = os.environ.get("ASCEND_HOME_PATH")
    executable = matrix.ROOT / "examples" / "build" / "s007_runner"
    if not home or not executable.is_file():
        raise RuntimeError("Initialize CANN and run bash build.sh -j8 -e first")
    vendor = matrix.ROOT / "build" / "packages" / "vendors" / "sync_batch_norm_gather_stats_custom"
    env = os.environ.copy()
    env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor) + os.pathsep + env.get("ASCEND_CUSTOM_OPP_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join((str(vendor / "op_api" / "lib"),
                                              str(Path(home) / "lib64"), env.get("LD_LIBRARY_PATH", "")))
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=True)
    results = {}
    for case, data in cases:
        for run in range(1, args.repeat + 1):
            run_case = dict(case, name=f"{case['name']}_run{run}")
            with mock.patch.object(matrix, "inputs_for", return_value=data):
                result = matrix.run_case(run_case, 0, base, executable, env)
            results[run_case["name"]] = result
            print(run_case["name"] + ": " + json.dumps(result), flush=True)
            (base / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Original channel indices:", INDICES.tolist(), flush=True)
    print("Local reference only: not an independent built-in or platform golden.", flush=True)
    return int(any(result["status"] != "PASS" for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
