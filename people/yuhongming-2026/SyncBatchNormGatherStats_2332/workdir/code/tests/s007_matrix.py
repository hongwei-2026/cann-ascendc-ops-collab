#!/usr/bin/env python3
"""Run reproducible 910B cases; W>1 and multi-count are diagnostic probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "samples"))
import reference  # noqa: E402

NAMES = ("batch_mean", "batch_invstd", "new_mean", "new_variance")
CHANNELS = (1, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
            127, 255, 256, 257, 513, 1023, 4097)


def cases():
    result = []
    for dtype in ("f32", "f16"):
        for count_type in ("i32", "i64"):
            for c in CHANNELS:
                result.append(dict(name=f"normal_{dtype}_{count_type}_c{c}",
                                   dtype=dtype, count_type=count_type, c=c,
                                   world=1, counts=[100], momentum=0.1,
                                   eps=1e-5, kind="normal", oracle=True))
            for kind, count, momentum, eps in (
                ("one", 1, 0.1, 1e-5),
                ("zero_var", 100, 0.1, 0.0),
                ("negative_var", 100, 0.1, 1e-5),
                ("momentum_zero", 100, 0.0, 1e-5),
                ("momentum_one", 100, 1.0, 1e-5),
                ("rounding", 100, 0.1, 1e-5),
            ):
                result.append(dict(name=f"{kind}_{dtype}_{count_type}", dtype=dtype,
                                   count_type=count_type, c=6, world=1,
                                   counts=[count], momentum=momentum, eps=eps,
                                   kind=kind, oracle=True))
            for count in (0, 2, 3, 7, 16, 257, 65535, 16777217):
                result.append(dict(name=f"count{count}_{dtype}_{count_type}", dtype=dtype,
                                   count_type=count_type, c=33, world=1, counts=[count],
                                   momentum=0.3, eps=0.001, kind="normal", oracle=True))
            result.append(dict(name=f"rank1_{dtype}_{count_type}", dtype=dtype, count_type=count_type,
                               c=17, world=1, counts=[100], momentum=0.1, eps=1e-5,
                               kind="normal", oracle=True, rank="rank1"))
            for world, counts in ((1, [50, 100]), (2, [100]), (2, [50, 100])):
                result.append(dict(name=f"probe_w{world}_n{len(counts)}_{dtype}_{count_type}",
                                   dtype=dtype, count_type=count_type, c=17,
                                   world=world, counts=counts, momentum=0.1,
                                   eps=1e-5, kind="normal", oracle=False))
    result.append(dict(name="large_f32_i32_c524289", dtype="f32", count_type="i32",
                       c=524289, world=1, counts=[100], momentum=0.1,
                       eps=1e-5, kind="normal", oracle=True))
    return result


def inputs_for(case, seed):
    rng = np.random.default_rng(seed)
    c, world = case["c"], case["world"]
    dtype = np.float16 if case["dtype"] == "f16" else np.float32
    count_dtype = np.int64 if case["count_type"] == "i64" else np.int32
    sums = rng.uniform(-80, 80, (world, c)).astype(dtype)
    squares = rng.uniform(1000, 5000, (world, c)).astype(dtype)
    mean = rng.uniform(-5, 5, c).astype(dtype)
    variance = rng.uniform(1, 10, c).astype(dtype)
    if case["kind"] == "one":
        sums[:] = np.array([1, 1, 0, 2, -1, 0], dtype=dtype)
        squares[:] = np.array([2, 0, 0, 4, 2, 1], dtype=dtype)
    elif case["kind"] in ("zero_var", "negative_var"):
        sums.fill(100)
        squares.fill(100 if case["kind"] == "zero_var" else 50)
    elif case["kind"] == "rounding":
        sums[:] = np.array([465.5, 442.5, 411.25, 465, 378, 100], dtype=dtype)
        squares[:] = np.array([4080, 2088, 2590, 3672, 3806, 100.001], dtype=dtype)
        mean.fill(3)
        variance[:] = np.array([6.375, 4.45703125, 7.65234375, 3.19921875, 4.734375, 1], dtype=dtype)
    return dict(total_sum=sums, total_square_sum=squares,
                sample_count=np.asarray(case["counts"], dtype=count_dtype),
                mean=mean, variance=variance)


def golden(case, data):
    return reference.impl(**data, momentum=case["momentum"], eps=case["eps"])


def probe_golden(case, data, world_mode, count_mode):
    q = reference._q16
    sums = q(data["total_sum"])
    squares = q(data["total_square_sum"])
    total_sum = sums[0].copy()
    total_sq = squares[0].copy()
    if world_mode == "sum":
        for i in range(1, case["world"]):
            total_sum = np.float32(total_sum + sums[i])
            total_sq = np.float32(total_sq + squares[i])
    n = np.float64(data["sample_count"][0] if count_mode == "first"
                   else np.sum(data["sample_count"], dtype=np.int64))
    mom = np.float64(case["momentum"])
    eps = np.float64(case["eps"])
    with np.errstate(all="ignore"):
        mean_all = np.float32(total_sum / np.float32(n))
        var = np.float32(np.float32(total_sq / np.float32(n)) - np.float32(mean_all * mean_all))
        inv = np.float32(np.float32(1.0) / np.sqrt(np.float32(var + np.float32(eps))))
        factor = np.float64(n) / (n - 1) if n != 1 else np.float64("inf")
        new_mean = np.float32(np.float32(1.0 - mom) * q(data["mean"]) + np.float32(mom) * mean_all)
        new_var = np.float32(factor * mom * np.float64(var) + np.float64(1.0 - mom) * q(data["variance"]))
    dtype = data["total_sum"].dtype
    return (q(mean_all).astype(dtype), q(inv).astype(dtype), q(new_mean), q(new_var))


def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"output shape/dtype {actual.shape}/{actual.dtype}, expected {expected.shape}/{expected.dtype}")
    close = np.isclose(actual.astype(np.float64), expected.astype(np.float64),
                       rtol=1e-4, atol=1e-4, equal_nan=True)
    indexes = np.flatnonzero(~close)
    return dict(mismatched=int(indexes.size), total=int(expected.size),
                examples=[dict(index=int(i), actual=str(actual[i]), expected=str(expected[i]))
                          for i in indexes[:8]])


def run_case(case, index, base, executable, env):
    directory = base / case["name"]
    for folder in ("input", "output", "golden"):
        (directory / folder).mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        (directory / "output" / (name + ".bin")).unlink(missing_ok=True)
        (directory / "golden" / (name + ".bin")).unlink(missing_ok=True)
    seed = 20261119 + index
    data = inputs_for(case, seed)
    for name, value in data.items():
        value.tofile(directory / "input" / (name + ".bin"))
    (directory / "case.json").write_text(json.dumps(dict(case, seed=seed), indent=2), encoding="utf-8")
    args = [str(executable), str(directory), env.get("DEVICE_ID", "0"), str(case["c"]),
            str(case["world"]), str(len(case["counts"])), case["dtype"], case["count_type"],
            str(case["momentum"]), str(case["eps"]), case.get("rank", "rank2")]
    try:
        proc = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=180, errors="replace")
        (directory / "runner.log").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode:
            return dict(status="ERROR", detail=f"runner exit {proc.returncode}: {proc.stdout[-1000:]}")
        output = []
        for i, name in enumerate(NAMES):
            dtype = (np.float16 if case["dtype"] == "f16" else np.float32) if i < 2 else np.float32
            value = np.fromfile(directory / "output" / (name + ".bin"), dtype=dtype)
            if value.size != case["c"]:
                raise ValueError(f"{name}: expected {case['c']} values, got {value.size}")
            output.append(value)
        if not case["oracle"]:
            alternatives = {}
            for world_mode in ("first", "sum"):
                for count_mode in ("first", "sum"):
                    label = f"world_{world_mode}_count_{count_mode}"
                    candidate = probe_golden(case, data, world_mode, count_mode)
                    alternatives[label] = {
                        name: compare(value, want) for name, value, want in zip(NAMES, output, candidate)
                    }
                    folder = directory / ("candidate_" + label)
                    folder.mkdir(exist_ok=True)
                    for name, value in zip(NAMES, candidate):
                        value.tofile(folder / (name + ".bin"))
            return dict(status="PROBE", detail="unvalidated semantic candidates, not platform golden",
                        candidates=alternatives)
        expected = golden(case, data)
        comparisons = {name: compare(value, want) for name, value, want in zip(NAMES, output, expected)}
        for name, value in zip(NAMES, expected):
            value.tofile(directory / "golden" / (name + ".bin"))
        return dict(status="FAIL" if any(x["mismatched"] for x in comparisons.values()) else "PASS",
                    outputs=comparisons)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return dict(status="ERROR", detail=str(error))


def select_cases(matrix, name):
    return [(i, case) for i, case in enumerate(matrix) if not name or case["name"] == name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "samples" / "build" / "s007")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--case", help="Run one named case using its original matrix seed")
    args = parser.parse_args()
    matrix = cases()
    if args.self_test:
        assert len({case["name"] for case in matrix}) == len(matrix)
        selected = select_cases(matrix, "large_f32_i32_c524289")
        assert len(selected) == 1 and selected[0][0] == len(matrix) - 1
        assert not select_cases(matrix, "nonexistent")
        for i, case in enumerate(matrix):
            data = inputs_for(case, 20261119 + i)
            assert data["total_sum"].shape == (case["world"], case["c"])
            assert data["sample_count"].size == len(case["counts"])
            if case["world"] == 1:
                expected = golden(case, data)
                assert all(compare(x, x)["mismatched"] == 0 for x in expected)
                candidate = probe_golden(case, data, "first", "first")
                assert all(compare(x, y)["mismatched"] == 0 for x, y in zip(candidate, expected))
            else:
                assert all(x.size == case["c"] for x in probe_golden(case, data, "sum", "sum"))
        assert compare(np.array([np.inf, -np.inf, np.nan]), np.array([np.inf, -np.inf, np.nan]))["mismatched"] == 0
        assert compare(np.array([np.inf, np.nan]), np.array([-np.inf, 0.0]))["mismatched"] == 2
        case = matrix[0]
        def fake_runner(cmd, **kwargs):
            folder = Path(cmd[1])
            output = golden(case, inputs_for(case, 20261119))
            for name, value in zip(NAMES, output):
                value.tofile(folder / "output" / (name + ".bin"))
            return subprocess.CompletedProcess(cmd, 0, stdout="")
        def wrong_runner(cmd, **kwargs):
            result = fake_runner(cmd, **kwargs)
            output_path = Path(cmd[1]) / "output" / "batch_mean.bin"
            bad = np.fromfile(output_path, dtype=np.float32)
            bad[0] += 10
            bad.tofile(output_path)
            return result
        with tempfile.TemporaryDirectory(prefix="s007-selftest-", dir=ROOT / "tests") as tmp:
            with mock.patch("subprocess.run", side_effect=fake_runner):
                result = run_case(case, 0, Path(tmp), Path("fake-runner"), {})
            assert result["status"] == "PASS"
            with mock.patch("subprocess.run", side_effect=wrong_runner):
                result = run_case(case, 0, Path(tmp), Path("fake-runner"), {})
            assert result["status"] == "FAIL" and result["outputs"]["batch_mean"]["mismatched"] == 1
            with mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], 2, stdout="fake error")):
                result = run_case(case, 0, Path(tmp), Path("fake-runner"), {})
            assert result["status"] == "ERROR"
        print(f"SELF-TEST PASS: {len(matrix)} generated cases; no device executed")
        return 0
    home = os.environ.get("ASCEND_HOME_PATH")
    executable = ROOT / "examples" / "build" / "s007_runner"
    if not home or not executable.is_file():
        raise RuntimeError("Initialize CANN and run bash build.sh -j8 -e first")
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=True)
    vendor = ROOT / "build" / "packages" / "vendors" / "sync_batch_norm_gather_stats_custom"
    env = os.environ.copy()
    env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor) + os.pathsep + env.get("ASCEND_CUSTOM_OPP_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join((str(vendor / "op_api" / "lib"),
                                               str(Path(home) / "lib64"), env.get("LD_LIBRARY_PATH", "")))
    selected = select_cases(matrix, args.case)
    if not selected:
        parser.error(f"unknown case: {args.case}")
    results = {}
    for position, (i, case) in enumerate(selected, 1):
        result = run_case(case, i, base, executable, env)
        results[case["name"]] = result
        print(f"{position}/{len(selected)} {case['name']}: {result['status']} "
              + json.dumps(result, ensure_ascii=False), flush=True)
        (base / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    totals = {key: sum(r["status"] == key for r in results.values())
              for key in ("PASS", "FAIL", "ERROR", "PROBE")}
    print("TOTAL " + json.dumps(totals) + " (PROBE results need semantic adjudication)", flush=True)
    return int(totals["FAIL"] > 0 or totals["ERROR"] > 0)


if __name__ == "__main__":
    sys.exit(main())
