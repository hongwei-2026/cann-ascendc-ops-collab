"""J054 plans and ordinary-profiler summary; Python standard library only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ORDERS = ("ABCA", "CAAB", "BACA", "ACBA", "BAAC", "CABA")
METRICS = ("Task Duration(us)", "aiv_time(us)", "aiv_scalar_time(us)",
           "aiv_vec_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)")
DTYPES = {"fp32": "FLOAT", "fp16": "FLOAT16", "bf16": "DT_BF16"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_rows(cases, mode):
    rows = []
    for case in cases:
        if mode == "perf":
            windows = []
            for rnd, order in enumerate(ORDERS, 1):
                count_a = 0
                for letter in order:
                    if letter == "A":
                        count_a += 1
                        label = "A" + str(count_a)
                    else:
                        label = letter
                    windows.append((rnd, label, 350, 0))
        else:
            windows = [(0, label, case["iterations"], 1) for label in ("A1", "B", "C")]
        for rnd, label, calls, check_every in windows:
            rows.append(dict(case=case["id"], input_root=case["input_root"],
                             dtype=case["dtype"], shape=case["shape"], wlen=case["wlen"],
                             round=rnd, label=label, iterations=calls, check_every=check_every))
    return rows


def prepare(root, out, mode):
    if mode == "perf":
        cases = json.loads((root / "J052-abc/inputs.json").read_text())
        require(len(cases) == 27, "expected the existing 27-case campaign")
        require(len({c["id"] for c in cases}) == 27, "duplicate case ids")
        for case in cases:
            for name, digest in case["hashes"].items():
                actual = hashlib.sha256((Path(case["input_root"]) / name).read_bytes()).hexdigest()
                require(actual == digest, "input hash changed: " + case["id"] + "/" + name)
        random.Random(53003).shuffle(cases)
    else:
        source = root / "J050-layout-campaign" / mode / "plan.txt"
        cases = []
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            require(len(fields) >= 5, "bad prior plan row")
            input_root, _, dtype, dims, wlen = fields[:5]
            calls = 1 if mode == "precision" else int(fields[5])
            cases.append(dict(id=str(len(cases)), input_root=input_root, dtype=dtype,
                              shape=list(map(int, dims.split(","))), wlen=int(wlen),
                              iterations=calls))
        require(len(cases) == (780 if mode == "precision" else 57), "prior case count changed")
        require(sum(c["iterations"] for c in cases) == (780 if mode == "precision" else 5184),
                "prior call count changed")
    rows = make_rows(cases, mode)
    for row in rows:
        require(row["dtype"] in DTYPES, "unsupported dtype")
        require(not any(ch.isspace() for ch in row["input_root"]), "plan paths cannot contain whitespace")
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "manifest.json", dict(mode=mode, cases=cases, rows=rows,
         orders=ORDERS, discard=200 if mode == "perf" else 0,
         case_order_seed=53003 if mode == "perf" else None,
         allocation="one case at a time; all 24 perf windows share five buffers and one stream"))
    lines = [" ".join(map(str, [r["label"], r["round"], r["input_root"], r["dtype"],
             ",".join(map(str, r["shape"])), r["wlen"], r["iterations"], r["check_every"]]))
             for r in rows]
    (out / "plan.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(mode, len(rows), "windows;", sum(r["iterations"] for r in rows), "calls")


def read_runner(log, rows):
    begun, ended, addresses = {}, {}, {}
    for line in log.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "CASE_BEGIN":
            require(len(words) == 13 and words[7] == "ADDR", "malformed CASE_BEGIN")
            idx = int(words[1])
            require(0 <= idx < len(rows) and idx not in begun, "unexpected/duplicate window")
            row = rows[idx]
            require(words[2:7] == [row["label"], str(row["round"]), row["dtype"],
                    ",".join(map(str, row["shape"])), str(row["iterations"])], "runner/plan mismatch")
            addr = words[8:]
            require(len(set(addr)) == 5 and all(a not in ("0", "0x0", "(nil)") for a in addr),
                    "invalid or overlapping tensor base addresses")
            if row["case"] in addresses:
                require(addresses[row["case"]] == addr, "addresses changed within a case")
            addresses[row["case"]] = addr
            begun[idx] = addr
        elif words[0] == "CASE_END":
            require(len(words) == 4 and words[2] == "PASS", "malformed CASE_END")
            idx = int(words[1])
            require(idx in begun and idx not in ended, "unexpected completion")
            expected = rows[idx]["iterations"] if rows[idx]["check_every"] else 1
            require(int(words[3]) == expected, "missing correctness checks")
            ended[idx] = True
    require(len(begun) == len(rows) == len(ended), "incomplete runner log")
    return begun


def analyze(manifest_path, runner_log, profile, output):
    manifest = json.loads(manifest_path.read_text())
    require(manifest["mode"] == "perf", "profiling analysis requires a perf plan")
    rows = manifest["rows"]
    addresses = read_runner(runner_log, rows)
    samples = []
    for file in sorted(profile.rglob("op_summary*.csv")):
        with file.open(encoding="utf-8-sig", newline="") as stream:
            samples.extend(r for r in csv.DictReader(stream)
                           if r["OP Type"].startswith("PReluGradUpdate"))
    samples.sort(key=lambda r: float(r["Task Start Time(us)"]))
    require(len(samples) == sum(r["iterations"] for r in rows), "kernel count mismatch")
    cases = {c["id"]: c for c in manifest["cases"]}
    windows, groups = [], {}
    offset = 0
    for ordinal, row in enumerate(rows):
        block = samples[offset:offset + row["iterations"]]
        offset += row["iterations"]
        for sample in block:
            require(sample["OP Type"] == "PReluGradUpdate" + row["label"][0], "alias/order mismatch")
            shape = ",".join(map(str, row["shape"]))
            require(sample["Input Shapes"].strip('"').split(";") == [shape, shape, str(row["wlen"])],
                    "profile shape mismatch")
            require(sample["Input Data Types"].split(";") == [DTYPES[row["dtype"]]] * 3,
                    "profile dtype mismatch")
        tiling = cases[row["case"]]["derived_tiling"]
        block_nums = sorted({int(s["Block Num"]) for s in block})
        if tiling["split_c"]:
            require(block_nums == [tiling["block_dim"]], "split-C block count mismatch")
        tail = block[manifest["discard"]:]
        require(len(tail) == 150, "expected 150 retained samples")
        record = dict(row, ordinal=ordinal, addresses=addresses[ordinal], block_nums=block_nums)
        for metric in METRICS:
            values = [float(s[metric]) for s in tail]
            require(all(math.isfinite(v) and v >= 0 for v in values), "invalid metric")
            record[metric] = dict(median=statistics.median(values), min=min(values), max=max(values))
        windows.append(record)
        group = groups.setdefault((row["case"], row["round"]), {})
        require(row["label"] not in group, "duplicate label within round")
        group[row["label"]] = record
    comparisons = []
    for (case_id, rnd), group in groups.items():
        require(set(group) == {"A1", "A2", "B", "C"}, "incomplete A/A/B/C round")
        t = {label: rec["Task Duration(us)"]["median"] for label, rec in group.items()}
        a = (t["A1"] + t["A2"]) / 2
        require(a > 0, "zero baseline duration")
        spread = abs(t["A1"] - t["A2"])
        case = cases[case_id]
        unchanged = not case["modified"]
        control_times = list(t.values()) if unchanged else [t["A1"], t["A2"], t["C"]]
        control_spread = max(control_times) - min(control_times)
        comparisons.append(dict(case=case_id, round=rnd, dtype=case["dtype"],
            shape=case["shape"], group=case.get("group"), median_us=t, aa_spread_us=spread,
            aa_spread_fraction=spread/a, aa_stable=spread <= 0.1 and spread/a <= 0.02,
            unchanged_control=unchanged, controls_stable=control_spread <= 0.1 and control_spread/a <= 0.02,
            c_gain_vs_b_us=t["B"]-t["C"], c_gain_vs_a_us=a-t["C"]))
    dump(output, dict(windows=windows, comparisons=comparisons,
         all_controls_stable=all(c["controls_stable"] for c in comparisons),
         status="measurement only; compiled function-body correspondence still requires separate evidence; no promotion"))
    print(len(windows), "windows;", len(comparisons), "comparisons; controls stable:",
          all(c["controls_stable"] for c in comparisons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("mode", choices=("precision", "stress", "perf"))
    p = commands.add_parser("analyze")
    p.add_argument("manifest", type=Path)
    p.add_argument("runner_log", type=Path)
    p.add_argument("profile", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.out, args.mode)
    else:
        analyze(args.manifest, args.runner_log, args.profile, args.output)


if __name__ == "__main__":
    main()
