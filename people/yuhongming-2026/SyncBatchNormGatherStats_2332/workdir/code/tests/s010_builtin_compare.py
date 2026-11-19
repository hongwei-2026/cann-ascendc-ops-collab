#!/usr/bin/env python3
"""Compare the S010 channels using the installed ACLNN interface."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import s007_matrix as matrix
from s010_compact import INDICES


def original_cases():
    index, original = matrix.select_cases(matrix.cases(), "large_f32_i32_c524289")[0]
    source = matrix.inputs_for(original, 20261119 + index)
    for dtype in ("f32", "f16"):
        for count_type in ("i32", "i64"):
            ft = np.float16 if dtype == "f16" else np.float32
            ct = np.int64 if count_type == "i64" else np.int32
            data = dict(total_sum=source["total_sum"][:, INDICES].astype(ft),
                        total_square_sum=source["total_square_sum"][:, INDICES].astype(ft),
                        sample_count=source["sample_count"].astype(ct),
                        mean=source["mean"][INDICES].astype(ft),
                        variance=source["variance"][INDICES].astype(ft))
            case = dict(original, dtype=dtype, count_type=count_type, c=len(INDICES))
            yield case, data


def compare_outputs(actual, expected):
    return {name: matrix.compare(a.astype(np.float32), b.astype(np.float32))
            for name, a, b in zip(matrix.NAMES, actual, expected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=matrix.ROOT / "samples" / "build" / "s010-builtin")
    parser.add_argument("--custom-output", type=Path, default=matrix.ROOT / "samples" / "build" / "s010")
    parser.add_argument("--runner", type=Path, default=matrix.ROOT / "samples" / "build" / "s010_builtin_runner")
    args = parser.parse_args()
    cases = list(original_cases())
    if args.self_test:
        assert len(cases) == 4
        for case, data in cases:
            expected = matrix.golden(case, data)
            assert all(x["mismatched"] == 0 for x in compare_outputs(expected, expected).values())
            assert all(x.size == len(INDICES) for x in expected)
            wrong = list(expected)
            wrong[0] = expected[0].copy()
            wrong[0][0] = np.float32(wrong[0][0]) + 1
            assert compare_outputs(wrong, expected)["batch_mean"]["mismatched"] == 1
        print("SELF-TEST PASS: four original inputs; no device executed")
        return 0
    home = os.environ.get("ASCEND_HOME_PATH")
    if not home or not args.runner.is_file():
        raise RuntimeError("Initialize CANN and compile s010_builtin_runner first")
    home = Path(home).resolve()
    env = os.environ.copy()
    for name in ("ASCEND_CUSTOM_OPP_PATH", "LD_PRELOAD", "LD_AUDIT"):
        env.pop(name, None)
    kept = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if p and "vendors" not in p.lower() and "workdir/code" not in p.replace("\\", "/").lower()]
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(home / "lib64")] + kept)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for case, data in cases:
        for mode in (("native", "fp16_input") if case["dtype"] == "f32" else ("native",)):
            label = f"{case['dtype']}_{case['count_type']}_{mode}"
            folder = args.output / label
            for part in ("input", "output"):
                (folder / part).mkdir(parents=True, exist_ok=True)
            for name in matrix.NAMES:
                (folder / "output" / (name + ".bin")).unlink(missing_ok=True)
            run_dtype = "f16" if mode == "fp16_input" else case["dtype"]
            input_data = {name: (value.astype(np.float16) if mode == "fp16_input" and
                                 name != "sample_count" else value) for name, value in data.items()}
            for name, value in input_data.items():
                value.tofile(folder / "input" / (name + ".bin"))
            cmd = [str(args.runner.resolve()), str(folder.resolve()), env.get("DEVICE_ID", "0"),
                   str(case["c"]), run_dtype, case["count_type"], str(case["momentum"]),
                   str(case["eps"]), str(case["world"])]
            try:
                proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, timeout=180, errors="replace")
                (folder / "runner.log").write_text(proc.stdout, encoding="utf-8")
                if proc.returncode:
                    raise RuntimeError(f"runner exit {proc.returncode}: {proc.stdout[-2000:]}")
                library = next((line.split(": ", 1)[1] for line in proc.stdout.splitlines()
                                if line.startswith("ACLNN implementation: ")), "")
                if not library or not Path(library).resolve().is_relative_to(home):
                    raise RuntimeError(f"ACLNN library is not under ASCEND_HOME_PATH: {library}")
                actual = [np.fromfile(folder / "output" / (name + ".bin"),
                                      dtype=np.float16 if run_dtype == "f16" else np.float32)
                          for name in matrix.NAMES]
                if any(x.size != case["c"] for x in actual):
                    raise ValueError("builtin output size mismatch")
                entry = dict(status="EXECUTED", library=library,
                             vs_reference=compare_outputs(actual, matrix.golden(case, data)))
                custom = args.custom_output / f"compact_{case['dtype']}_{case['count_type']}_run1"
                if all((custom / "input" / (name + ".bin")).is_file() and
                       (custom / "input" / (name + ".bin")).read_bytes() == value.tobytes()
                       for name, value in data.items()):
                    custom_values = [np.fromfile(custom / "output" / (name + ".bin"),
                                     dtype=(np.float16 if case["dtype"] == "f16" and i < 2
                                            else np.float32)) for i, name in enumerate(matrix.NAMES)]
                    entry["vs_custom"] = compare_outputs(actual, custom_values)
                else:
                    entry["vs_custom"] = "SKIP: custom inputs missing or not byte-identical"
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                entry = dict(status="ERROR", detail=str(error))
            results[label] = entry
            print(label + ": " + json.dumps(entry), flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Library isolation does not alone prove which device kernel was dispatched.", flush=True)
    return int(any(value["status"] == "ERROR" for value in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
