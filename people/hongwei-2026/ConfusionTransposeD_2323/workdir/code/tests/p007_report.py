#!/usr/bin/env python3
"""Print compact evidence from an existing P007 run, including failed stages."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def read_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    if (root / "logs").is_dir():
        root /= "logs"
    print("C007_EVIDENCE_BEGIN")
    for name in ("baseline_commit.txt", "candidate_commit.txt", "exit_code.txt", "stage.txt", "environment.log"):
        path = root / name
        print(f"FILE {name}")
        print(path.read_text(errors="replace")[:4000] if path.is_file() else "MISSING")
    print("Same-machine P004/P007 custom kernel comparison; platform Runtime Error remains unresolved.")
    profiles = {}
    for label in ("baseline", "candidate"):
        results = read_json(root / f"{label}_regression/results.json", [])
        print(label, "REGRESSION_COUNTS", dict(Counter(r["result"] for r in results)))
        for record in results:
            if record["result"] != "PASS_BITWISE":
                print("NONPASS", json.dumps(record))
        rows = read_json(root / f"{label}_profile/profile_summary.json", [])
        profiles[label] = {r.get("case"): r for r in rows}
        for record in rows:
            print(label, "PROFILE", json.dumps(record))
    for name, new in profiles["candidate"].items():
        old = profiles["baseline"].get(name, {})
        if old.get("result") == new.get("result") == "PROFILED":
            print("COMPARISON", json.dumps(dict(case=name, baseline_us=old["median_us"],
                candidate_us=new["median_us"], speedup=old["median_us"] / new["median_us"])))
    error_pattern = re.compile(r"\[ERROR\]|ACL error|error:|TIMEOUT|Traceback|FAILED|mismatch(?!ed=0)", re.I)
    for path in sorted(root.rglob("*.log")):
        if path.name in ("run_full.log", "terminal_report.log"):
            continue
        matches = []
        with path.open(errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if error_pattern.search(line):
                    matches.append(f"{number}: {line.rstrip()}")
        if matches:
            raw = path.read_bytes()
            print(f"LOG {path.relative_to(root)} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}")
            selected = matches if len(matches) <= 6 else matches[:3] + [f"[omitted {len(matches)-6} matching lines]"] + matches[-3:]
            for line in selected:
                print(line[:1200])
    print("C007_EVIDENCE_END")


if __name__ == "__main__":
    main()
