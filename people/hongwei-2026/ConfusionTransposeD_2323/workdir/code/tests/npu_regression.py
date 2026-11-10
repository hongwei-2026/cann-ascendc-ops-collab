#!/usr/bin/env python3
"""UB-safe NPU generalization cases for ConfusionTransposeD (P002 / EXP_C002).

After `bash build.sh -e` (or at least build + example binary), run:
  python3 -B tests/npu_regression.py
Use --generate-only on CPU hosts without claiming kernel PASS.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "samples"))
import reference  # noqa: E402

try:
    from ml_dtypes import bfloat16 as bf16
except ImportError:  # pragma: no cover
    bf16 = None

# name, in_shape, perm, attr_shape, transpose_first, dtype_name
# Keep numel*bytes well under typical 910B UB (~192KB) with headroom for out tile.
CASES = [
    ("baseline_tf_f32", (2, 3, 4, 5), (0, 2, 1, 3), (2, 4, 3, 5), True, "float32"),
    ("baseline_rf_f32", (2, 3, 4, 5), (0, 2, 1, 3), (2, 4, 3, 5), False, "float32"),
    ("baseline_tf_f16", (2, 3, 4, 5), (0, 2, 1, 3), (2, 4, 3, 5), True, "float16"),
    ("baseline_tf_i8", (2, 3, 4, 5), (0, 2, 1, 3), (2, 4, 3, 5), True, "int8"),
    ("baseline_tf_i32", (2, 3, 4, 5), (0, 2, 1, 3), (2, 4, 3, 5), True, "int32"),
    ("rank2_swap_f32", (8, 16), (1, 0), (16, 8), True, "float32"),
    ("rank3_rot_f32", (4, 5, 6), (2, 0, 1), (6, 4, 5), True, "float32"),
    ("rank3_rf_f32", (4, 5, 6), (2, 0, 1), (6, 4, 5), False, "float32"),
    # tiling 要求 len(perm)==len(shape)；同 rank reshape（非仅 permute）
    ("reshape_same_rank_tf", (2, 3, 4, 5), (0, 2, 1, 3), (4, 3, 2, 5), True, "float32"),
    ("reshape_same_rank_rf", (2, 3, 4, 5), (0, 2, 1, 3), (4, 5, 3, 2), False, "float32"),
    ("larger_4d_f32", (4, 8, 8, 8), (0, 2, 1, 3), (4, 8, 8, 8), True, "float32"),
    ("larger_4d_f16", (4, 16, 16, 8), (0, 2, 1, 3), (4, 16, 16, 8), True, "float16"),
]

if bf16 is not None:
    CASES.append(("baseline_tf_bf16", (2, 3, 4, 5), (0, 2, 1, 3), (2, 4, 3, 5), True, "bfloat16"))

DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "int8": np.int8,
    "int32": np.int32,
    "bfloat16": bf16,
}

TOL = {
    "float32": (1e-4, 1e-4),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1e-2, 1e-2),
    "int8": (0.0, 0.0),
    "int32": (0.0, 0.0),
}


def final_shape(attr_shape, perm, transpose_first):
    if transpose_first:
        return list(attr_shape)
    return [attr_shape[p] for p in perm]


def write_meta(folder: Path, dtype, transpose_first, in_shape, perm, attr_shape):
    text = (
        f"dtype={dtype}\n"
        f"transpose_first={1 if transpose_first else 0}\n"
        f"in_shape={','.join(map(str, in_shape))}\n"
        f"perm={','.join(map(str, perm))}\n"
        f"out_shape={','.join(map(str, attr_shape))}\n"
    )
    (folder / "meta.txt").write_text(text, encoding="utf-8")


def make_input(rng, in_shape, dtype_name):
    dtype = DTYPE_MAP[dtype_name]
    if np.issubdtype(np.dtype(dtype), np.integer):
        return rng.integers(-8, 8, size=in_shape, dtype=dtype)
    return rng.uniform(-5, 5, size=in_shape).astype(dtype)


def compare(actual: np.ndarray, golden: np.ndarray, dtype_name: str) -> tuple[bool, int]:
    rtol, atol = TOL[dtype_name]
    if rtol == 0 and atol == 0:
        bad = int(np.count_nonzero(actual != golden))
        return bad == 0, bad
    close = np.isclose(
        actual.astype(np.float64),
        golden.astype(np.float64),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    bad = int(np.count_nonzero(~close))
    return bad == 0, bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--directory", type=Path, default=ROOT / "samples" / "build-c002-regression")
    parser.add_argument("--binary", type=Path, default=ROOT / "examples" / "build" / "sample")
    parser.add_argument("--device", default=os.environ.get("DEVICE_ID", "0"))
    args = parser.parse_args()
    if not args.generate_only and not args.binary.is_file():
        parser.error("Build the NPU example first (DEVICE_ID=0 bash build.sh -j4 -e)")

    rng = np.random.default_rng(2323002)
    env = os.environ.copy()
    vendor = ROOT / "build/packages/vendors/confusion_transpose_d_custom"
    env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor) + (
        ":" + env["ASCEND_CUSTOM_OPP_PATH"] if env.get("ASCEND_CUSTOM_OPP_PATH") else ""
    )
    env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")

    results = []
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, in_shape, perm, attr_shape, tf, dtype_name in CASES:
        if dtype_name not in DTYPE_MAP or DTYPE_MAP[dtype_name] is None:
            results.append({"case": name, "result": "SKIP_NO_DTYPE"})
            print("SKIP", name, "dtype unavailable")
            continue
        folder = args.directory / name
        for sub in ("input", "golden", "output"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        x = make_input(rng, in_shape, dtype_name)
        y = reference.impl(x, list(perm), list(attr_shape), tf)
        assert list(y.shape) == final_shape(attr_shape, perm, tf)
        (folder / "input" / "x.bin").write_bytes(x.tobytes())
        (folder / "golden" / "y.bin").write_bytes(np.ascontiguousarray(y).tobytes())
        write_meta(folder, dtype_name, tf, in_shape, perm, attr_shape)
        record = {
            "case": name,
            "dtype": dtype_name,
            "numel": int(x.size),
            "transpose_first": tf,
            "result": "GENERATED_ONLY",
        }
        if not args.generate_only:
            out_path = folder / "output" / "y.bin"
            out_path.unlink(missing_ok=True)
            proc = subprocess.run(
                [str(args.binary.resolve()), str(folder.resolve()), str(args.device)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
            )
            (folder / "run.log").write_text(proc.stdout, encoding="utf-8")
            ok = proc.returncode == 0
            bad = -1
            if ok and out_path.is_file() and out_path.stat().st_size == y.nbytes:
                actual = np.frombuffer(out_path.read_bytes(), dtype=y.dtype).reshape(y.shape)
                ok, bad = compare(actual, y, dtype_name)
            else:
                ok = False
            record["result"] = "PASS" if ok else "FAIL"
            record["mismatched"] = bad
            print(record["result"], name, f"mismatched={bad}/{y.size}", flush=True)
        results.append(record)

    (args.directory / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.generate_only:
        print("generated", len(results), "cases under", args.directory)
        return 0
    failed = [r for r in results if r["result"] == "FAIL"]
    print("summary", {r["result"]: sum(1 for x in results if x["result"] == r["result"]) for r in results})
    # clearer summary
    from collections import Counter

    print("counts", dict(Counter(r["result"] for r in results)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
