#!/usr/bin/env python3
"""Create a separate S008-derived project with one arithmetic change."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
KERNEL = Path("op_kernel/sync_batch_norm_gather_stats.h")
BASELINE_BLOB = "570c1b4c94278b1b4d9bc01022cdb8c53c7d1170"
OLD = """    const float unbiasMom = unbias * momentum_;
"""
OLD_MUL = """    Muls(load, accSq, unbiasMom, calcCount);
"""
NEW_MUL = """    Muls(load, accSq, unbias, calcCount);
    Muls(load, load, momentum_, calcCount);
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    target = args.destination.resolve()
    allowed = (ROOT / "samples" / "build").resolve()
    if not target.is_relative_to(allowed) or target == allowed:
        raise ValueError("destination must be a new directory under samples/build")
    if target.exists():
        raise FileExistsError(f"destination already exists: {target}")
    source = ROOT / KERNEL
    blob = subprocess.check_output(["git", "hash-object", str(source)], text=True).strip()
    if blob != BASELINE_BLOB:
        raise RuntimeError(f"unexpected kernel baseline {blob}; expected {BASELINE_BLOB}")
    content = source.read_text(encoding="utf-8")
    if content.count(OLD) != 1 or content.count(OLD_MUL) != 1:
        raise RuntimeError("expected kernel operations missing or ambiguous")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns("build", "build_out", "__pycache__", ".git"))
    variant = content.replace(OLD, "").replace(OLD_MUL, NEW_MUL)
    (target / KERNEL).write_text(variant, encoding="utf-8", newline="\n")
    print(f"S008 git blob: {blob}")
    print(f"Candidate kernel SHA256: {hashlib.sha256((target / KERNEL).read_bytes()).hexdigest()}")
    print(f"Candidate project: {target}")
    print("Single change: variance * unbias, then * momentum.")


if __name__ == "__main__":
    main()
