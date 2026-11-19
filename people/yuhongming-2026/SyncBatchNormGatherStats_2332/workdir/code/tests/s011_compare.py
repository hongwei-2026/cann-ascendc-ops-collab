#!/usr/bin/env python3
"""Compare S011 candidate, S008 baseline and saved system outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import s007_matrix as matrix


def load_outputs(folder, dtype, system=False):
    values = []
    for i, name in enumerate(matrix.NAMES):
        value_type = np.float16 if dtype == "f16" and (system or i < 2) else np.float32
        value = np.fromfile(folder / "output" / (name + ".bin"), dtype=value_type)
        if value.size != 15:
            raise ValueError(f"{folder}: {name} has {value.size} elements")
        values.append(value.astype(np.float32))
    return values


def compare(actual, expected):
    return {name: matrix.compare(a, b) for name, a, b in zip(matrix.NAMES, actual, expected)}


def ensure_same_inputs(first, second):
    for name in ("total_sum", "total_square_sum", "sample_count", "mean", "variance"):
        if (first / "input" / (name + ".bin")).read_bytes() != (second / "input" / (name + ".bin")).read_bytes():
            raise ValueError(f"input {name} differs: {first} vs {second}")


def ensure_system_inputs(case, system, dtype):
    for name in ("total_sum", "total_square_sum", "sample_count", "mean", "variance"):
        data = (case / "input" / (name + ".bin")).read_bytes()
        if name != "sample_count" and dtype == "f32":
            data = np.frombuffer(data, dtype=np.float32).astype(np.float16).tobytes()
        if data != (system / "input" / (name + ".bin")).read_bytes():
            raise ValueError(f"system input {name} differs: {case} vs {system}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    error = False
    for dtype in ("f32", "f16"):
        for count in ("i32", "i64"):
            for run in range(1, 4):
                name = f"compact_{dtype}_{count}_run{run}"
                baseline, candidate = args.baseline / name, args.candidate / name
                try:
                    ensure_same_inputs(baseline, candidate)
                    before = load_outputs(baseline, dtype)
                    after = load_outputs(candidate, dtype)
                    reference = []
                    for item in matrix.NAMES:
                        if ((candidate / "golden" / (item + ".bin")).read_bytes() !=
                                (baseline / "golden" / (item + ".bin")).read_bytes()):
                            raise ValueError(f"reference differs for {item}")
                        data = np.fromfile(candidate / "golden" / (item + ".bin"),
                                           dtype=np.float16 if dtype == "f16" and item in matrix.NAMES[:2]
                                           else np.float32)
                        if data.size != 15:
                            raise ValueError(f"reference size mismatch: {item}")
                        reference.append(data.astype(np.float32))
                    entry = dict(status="COMPARED", baseline_vs_reference=compare(before, reference),
                                 candidate_vs_reference=compare(after, reference),
                                 candidate_vs_baseline=compare(after, before))
                    entry["unaffected_outputs_exact"] = {
                        matrix.NAMES[i]: bool(np.array_equal(before[i], after[i], equal_nan=True))
                        for i in range(3)
                    }
                    if not all(entry["unaffected_outputs_exact"].values()):
                        entry["status"] = "ERROR_UNEXPECTED_CHANGE"
                        error = True
                    if count == "i32":
                        system = args.system / ("f32_i32_fp16_input" if dtype == "f32" else "f16_i32_native")
                        ensure_system_inputs(candidate, system, dtype)
                        expected = load_outputs(system, "f16", system=True)
                        entry["baseline_vs_system"] = compare(before, expected)
                        entry["candidate_vs_system"] = compare(after, expected)
                    else:
                        entry["system"] = "SKIP: installed system API rejected int64 count"
                except (OSError, ValueError) as exc:
                    entry = dict(status="ERROR", detail=str(exc))
                    error = True
                results[name] = entry
                print(name + ": " + json.dumps(entry), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("System library was isolated in S010; device kernel dispatch remains unverified.")
    return int(error)


if __name__ == "__main__":
    raise SystemExit(main())
