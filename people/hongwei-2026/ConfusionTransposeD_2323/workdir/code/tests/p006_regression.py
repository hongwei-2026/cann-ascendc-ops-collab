#!/usr/bin/env python3
"""Bitwise NPU probes for P006 strided DMA; --generate-only is never NPU PASS."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from npu_regression import CASES, P004_CASES, DTYPE_MAP, ROOT, reference, write_meta


STRIDED_CASES = [
    ("stride_large_f32", (128, 256), (1, 0), (256, 128), True, "float32"),
    ("stride_large_f16", (256, 256), (1, 0), (256, 256), True, "float16"),
    ("stride_large_i8", (256, 512), (1, 0), (512, 256), True, "int8"),
    ("stride_large_i64", (128, 128), (1, 0), (128, 128), True, "int64"),
    ("stride_multitile_i8", (128, 1407), (1, 0), (1407, 128), True, "int8"),
    ("stride_odd_f16", (255, 257), (1, 0), (257, 255), True, "float16"),
    ("stride_tail_f32", (3, 5), (1, 0), (5, 3), True, "float32"),
    ("stride_tail_i8", (7, 9), (1, 0), (9, 7), True, "int8"),
    ("stride_tail_f16", (17, 19), (1, 0), (19, 17), True, "float16"),
    ("stride_tail_i64", (5, 7), (1, 0), (7, 5), True, "int64"),
    ("stride_tail_bf16", (17, 19), (1, 0), (19, 17), True, "bfloat16"),
    ("stride_last_unit", (3, 5, 1), (1, 0, 2), (5, 3, 1), True, "float16"),
    ("stride_4d", (4, 16, 8, 8), (0, 2, 3, 1), (4, 8, 8, 16), True, "float32"),
    ("stride_reshape_tf", (4, 5, 6), (2, 0, 1), (3, 4, 10), True, "float32"),
    ("stride_reshape_rf", (4, 5, 6), (1, 2, 0), (3, 4, 10), False, "float32"),
    ("stride_crossrank_rf", (128, 64), (2, 0, 1), (16, 16, 32), False, "float16"),
    ("stride_rank8", (2, 1, 3, 1, 2, 1, 5, 1), (6, 1, 2, 3, 4, 5, 0, 7),
     (5, 1, 3, 1, 2, 1, 2, 1), True, "float32"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("DEVICE_ID", "0"))
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    binary = args.code_root.resolve() / "examples/build/sample"
    env = os.environ.copy()
    vendor = args.code_root.resolve() / "build/packages/vendors/confusion_transpose_d_custom"
    env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
    env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(CTD_PERF_WARMUP="0", CTD_PERF_REPEAT="1", ASCEND_GLOBAL_LOG_LEVEL="3", ASCEND_SLOG_PRINT_TO_STDOUT="1")
    rng = np.random.default_rng(2323006)
    records = []
    stop = False
    for name, shape, perm, attr, tf, dtype_name in CASES + P004_CASES + STRIDED_CASES:
        folder = args.directory / name
        for sub in ("input", "golden", "output"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        record = dict(case=name, in_shape=shape, perm=perm, attr_shape=attr, transpose_first=tf, dtype=dtype_name)
        dtype = DTYPE_MAP[dtype_name]
        if dtype is None:
            record["result"] = "SKIP_NO_DTYPE"
        elif stop:
            record["result"] = "SKIP_AFTER_DEVICE_ERROR"
        else:
            dtype = np.dtype(dtype)
            raw = rng.integers(0, 256, size=int(np.prod(shape)) * dtype.itemsize, dtype=np.uint8)
            x = raw.view(dtype).reshape(shape)
            golden = np.ascontiguousarray(reference.impl(x, list(perm), list(attr), tf)).tobytes()
            (folder / "input/x.bin").write_bytes(x.tobytes())
            (folder / "golden/y.bin").write_bytes(golden)
            write_meta(folder, dtype_name, tf, shape, perm, attr)
            record.update(input_bytes=x.nbytes, golden_sha256=hashlib.sha256(golden).hexdigest(), result="GENERATED_ONLY")
            output = folder / "output/y.bin"
            output.unlink(missing_ok=True)
            if not args.generate_only:
                try:
                    proc = subprocess.run([str(binary), str(folder.resolve()), args.device], env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                    (folder / "run.log").write_bytes(proc.stdout)
                    record.update(exit_code=proc.returncode, result="PASS_BITWISE" if proc.returncode == 0
                                  and output.is_file() and output.read_bytes() == golden else "FAIL")
                    stop = proc.returncode < 0
                except subprocess.TimeoutExpired as error:
                    (folder / "run.log").write_bytes((error.stdout or b"") + b"\nTIMEOUT after 60s\n")
                    record["result"] = "TIMEOUT"
                    stop = True
        records.append(record)
        (args.directory / "results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(name, record["result"], flush=True)
    counts = dict(Counter(r["result"] for r in records))
    print("P006_COUNTS", json.dumps(counts), flush=True)
    if args.generate_only:
        return 0 if all(r["result"] == "GENERATED_ONLY" for r in records) else 2
    return 0 if all(r["result"] == "PASS_BITWISE" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
