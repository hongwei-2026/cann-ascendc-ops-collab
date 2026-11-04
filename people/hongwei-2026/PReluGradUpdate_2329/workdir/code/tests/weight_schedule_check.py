#!/usr/bin/env python3
"""Check P105's exact source reordering; this is not a device performance test."""

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RELATIVE_KERNEL = "people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update.h"
COMPUTE = "__aicore__ inline void PReluGradUpdate<T>::Compute("
DEQUEUE = "    LocalTensor<T> gradsLocal = gradsQueue_.DeQue<T>();"
WEIGHT = "    const int64_t globalStart = blockStart_ + progress;"
SYNC = "    SyncSV();\n    Muls("


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    candidate = (ROOT / "op_kernel" / "p_relu_grad_update.h").read_text(encoding="utf-8")
    if args.baseline:
        baseline = args.baseline.read_text(encoding="utf-8")
    else:
        baseline = subprocess.check_output(
            ["git", "show", f"47db93e:{RELATIVE_KERNEL}"], cwd=ROOT
        ).decode("utf-8")
    begin = baseline.index(COMPUTE)
    entry = baseline.index(DEQUEUE, begin)
    start = baseline.index(WEIGHT, entry)
    end = baseline.index(SYNC, start)
    block = baseline[start:end]
    assert all(name not in block for name in ("gradsLocal", "featuresLocal", "gradsFloat", "featuresFloat"))
    shortened = baseline[:start] + baseline[end:]
    expected = shortened[:entry] + block + "\n" + shortened[entry:]
    assert candidate == expected, "P105 must only move the unchanged scalar weight block"
    compute = candidate[candidate.index(COMPUTE):]
    assert compute.index(WEIGHT) < compute.index(DEQUEUE) < compute.index(SYNC)
    process = candidate[candidate.index("__aicore__ inline void PReluGradUpdate<T>::Process()"):]
    assert process.index("CopyIn(progress, currentNum);") < process.index("Compute(progress, currentNum);")
    print("P105 exact source reordering: PASS (all calculations, barriers, allocation and copies unchanged)")


if __name__ == "__main__":
    main()
