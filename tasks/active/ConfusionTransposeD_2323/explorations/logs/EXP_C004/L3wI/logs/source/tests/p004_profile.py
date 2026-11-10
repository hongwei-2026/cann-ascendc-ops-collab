#!/usr/bin/env python3
"""Collect raw msprof kernel timings. CPU wall time is never a kernel result."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess


PROBES = ("baseline_tf_f32", "larger_4d_f16", "dma_rows128_f32", "rank2_swap_f32")


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def read_durations(folder):
    values, sources, blocks = [], [], []
    for path in sorted(folder.rglob("op_summary*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not any("confusiontransposed" in normalized(str(v)) for v in row.values()):
                    continue
                duration = next((v for k, v in row.items()
                                 if normalized(k) in ("taskdurationus", "durationus")), None)
                if duration is None:
                    continue
                values.append(float(duration))
                blocks.extend(v for k, v in row.items() if normalized(k) in ("blockdim", "blockdims"))
                sources.append(str(path.relative_to(folder)))
    return values, sorted(set(sources)), sorted(set(blocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("DEVICE_ID", "0"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profiler = shutil.which("msprof")
    records = []
    if profiler is None:
        records.append({"result": "NO_MSPROF", "kernel_timing": "UNVERIFIED"})
    else:
        root = args.code_root.resolve()
        vendor = root / "build/packages/vendors/confusion_transpose_d_custom"
        env = os.environ.copy()
        env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
        env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        env["CTD_PERF_WARMUP"] = str(args.warmup)
        env["CTD_PERF_REPEAT"] = str(args.repeat)
        for name in PROBES:
            case = (args.cases / name).resolve()
            output = args.output / name
            output.mkdir()
            (case / "output/y.bin").unlink(missing_ok=True)
            application = f"{root / 'examples/build/sample'} {case} {args.device}"
            command = [profiler, f"--application={application}", f"--output={output.resolve()}",
                       "--task-time=on", "--aic-metrics=PipeUtilization"]
            record = {"case": name, "command": command, "result": "UNVERIFIED"}
            try:
                proc = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, timeout=300)
                (output / "msprof_full.log").write_text(proc.stdout, encoding="utf-8")
                durations, sources, blocks = read_durations(output)
                actual = case / "output/y.bin"
                exact = actual.is_file() and actual.read_bytes() == (case / "golden/y.bin").read_bytes()
                record.update(exit_code=proc.returncode, exact_output=exact,
                              observed_kernel_calls=len(durations), csv_files=sources, block_dim=blocks)
                expected_calls = args.warmup + args.repeat
                if (proc.returncode == 0 and exact and len(durations) == expected_calls
                        and all(math.isfinite(v) and v > 0 for v in durations)):
                    measured = durations[args.warmup:]
                    record.update(result="PROFILED", warmup=args.warmup, samples_us=measured,
                                  median_us=statistics.median(measured), min_us=min(measured), max_us=max(measured))
                else:
                    record["reason"] = "Need successful execution, exact bytes and one identified kernel per call"
            except (subprocess.TimeoutExpired, OSError, ValueError) as error:
                if isinstance(error, subprocess.TimeoutExpired):
                    captured = error.stdout or b""
                    if isinstance(captured, bytes):
                        captured = captured.decode(errors="replace")
                    (output / "msprof_full.log").write_text(captured, encoding="utf-8")
                record.update(result="PROFILE_ERROR", reason=str(error))
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    (args.output / "profile_summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return 0 if len(records) == len(PROBES) and all(r["result"] == "PROFILED" for r in records) else 3


if __name__ == "__main__":
    raise SystemExit(main())
