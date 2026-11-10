#!/usr/bin/env python3
"""Print existing C005 evidence for copying from a terminal; never run an NPU case."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import re


def show_errors(path):
    """Scan the whole file; retain bounded error windows rather than its ends."""
    print(f"\nFILE {path}")
    if not path.is_file():
        print("MISSING")
        return
    rules = {
        "operator_reason": re.compile(
            r"tensor too large for UB|perm/shape rank mismatch|shape numel mismatch|"
            r"perm out of range|invalid input rank|empty tensor|^ACL error\b|"
            r"ConfusionTranspose.*(?:fail|error|mismatch)", re.I),
        "error": re.compile(r"\[(?:ERROR|FATAL)\]|\b(?:ERROR|FATAL)\s*[:(]|"
                            r"\b(?:Tiling|InferShape).*\bfailed\b", re.I),
        "phase": re.compile(r"^DIAG\b"),
    }
    first = {name: [] for name in rules}
    last = {name: deque(maxlen=3) for name in rules}
    counts = Counter()
    pending = []
    previous = None
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for number, raw in enumerate(source, 1):
            digest.update(raw)
            size += len(raw)
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            for window in pending:
                window.append((number, line))
            pending = []
            for name, rule in rules.items():
                if not rule.search(line):
                    continue
                counts[name] += 1
                window = ([] if previous is None else [previous]) + [(number, line)]
                if len(first[name]) < 3:
                    first[name].append(window)
                else:
                    last[name].append(window)
                pending.append(window)
            previous = (number, line)
    print(f"bytes={size} sha256={digest.hexdigest()}")
    print("FILTERED_EXCERPT: whole file scanned; original line numbers retained; not a full log.")
    for name in rules:
        windows = first[name] + list(last[name])
        print(f"MATCHES {name} total={counts[name]} retained={len(windows)} omitted={counts[name] - len(windows)}")
        lines = {number: line for window in windows for number, line in window}
        for number in sorted(lines):
            line = lines[number]
            if len(line) > 1000:
                line = line[:600] + " [LINE EXCERPT] " + line[-400:]
            print(f"{number}: {line}")
    if not any(counts.values()):
        print("NO_ERROR_OR_PHASE_MATCH; inspect this case with --full if needed.")


def show_file(path, limit):
    print(f"\nFILE {path}")
    if not path.is_file():
        print("MISSING")
        return
    raw = path.read_bytes()
    print(f"bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}")
    content = raw.decode("utf-8", errors="replace")
    if limit and len(content) > limit:
        half = limit // 2
        print(content[:half])
        print(f"\n[EXCERPT: omitted {len(content) - 2 * half} characters; --full prints the whole file]\n")
        print(content[-half:])
    else:
        print(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Existing EXP_C005 run directory or its logs directory")
    parser.add_argument("--version", choices=("baseline", "candidate", "both"), default="candidate")
    parser.add_argument("--case", help="Print only this case's details (including a passing case)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Print complete selected case logs")
    mode.add_argument("--errors", action="store_true", help="Scan whole case logs for errors, operator reasons, and phase markers")
    args = parser.parse_args()
    root = args.run.resolve()
    if (root / "logs").is_dir():
        root /= "logs"
    print("C005_TEXT_EVIDENCE_BEGIN")
    print("Read-only export; no build, device execution, or new acceptance result.")
    for name in ("diagnostic_commit.txt", "diagnostic_exit_code.txt"):
        show_file(root / name, 2000)
    versions = ("baseline", "candidate") if args.version == "both" else (args.version,)
    found_case = False
    complete = True
    for version in ("baseline", "candidate"):
        result_path = root / version / "results.json"
        if not result_path.is_file():
            print(f"MISSING {result_path}")
            complete = False
            continue
        raw = result_path.read_bytes()
        records = json.loads(raw)
        print(f"\nVERSION {version} cases={len(records)} results_sha256={hashlib.sha256(raw).hexdigest()}")
        print("COUNTS", json.dumps(dict(Counter(r["result"] for r in records)), sort_keys=True))
        for record in records:
            if record["result"] != "PASS_BITWISE":
                print("NONPASS", json.dumps(record, ensure_ascii=False, sort_keys=True))
        if version not in versions:
            continue
        for record in records:
            name = record["case"]
            selected = name == args.case if args.case else record["result"] != "PASS_BITWISE"
            if selected:
                found_case = True
                if Path(name).name != name:
                    raise ValueError("Invalid case name in stored results")
                path = root / version / name / "run_full.log"
                if args.errors:
                    show_errors(path)
                else:
                    show_file(path, 0 if args.full else 6000)
            if name in ("baseline_tf_f32", "rank2_swap_f32", "dma_rows128_f32", "empty_tensor"):
                print("HOST_TIMING", json.dumps({k: record.get(k) for k in (
                    "case", "result", "exit_code", "host_phases", "timing_scope"
                )}, ensure_ascii=False))
    if args.case and not found_case:
        print("REQUESTED_CASE_NOT_FOUND")
        complete = False
    print("C005_TEXT_EVIDENCE_END")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
