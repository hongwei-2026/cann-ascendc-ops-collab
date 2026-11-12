#!/usr/bin/env python3
"""Inspect tiling output metadata and report infer events separately."""
import json
import sys
from pathlib import Path


def records(log):
    events = []
    if not log.is_file():
        return events
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "CTD_PROBE {" in line:
            try:
                events.append(json.loads(line[line.index("CTD_PROBE {") + len("CTD_PROBE "):]))
            except json.JSONDecodeError:
                events.append({"parse_error": line})
    return events


def main(root):
    root = Path(root)
    result_file = root / "candidate_regression" / "results.json"
    if not result_file.exists():
        print("C009_PROBE_CHECK: missing regression results")
        return 2
    cases = json.loads(result_file.read_text(encoding="utf-8"))
    failures = []
    samples = []
    infer_unobserved = []
    for case in cases:
        name = case["case"]
        events = records(root / "candidate_regression" / name / "run.log")
        success = [e for e in events if e.get("phase") == "tiling" and e.get("event") == "exit"
                   and e.get("host_ok") is True]
        expected = (case["in_shape"], case["perm"], case["attr_shape"], case["transpose_first"])
        expected_output = (case["attr_shape"] if case["transpose_first"] else
                           [case["attr_shape"][i] for i in case["perm"]])
        matches = [e for e in success if (e.get("in_shape"), e.get("perm"), e.get("attr_shape"),
                                          e.get("transpose_first")) == expected
                   and isinstance(e.get("dtype_code"), int) and e.get("copy_span", 0) > 0
                   and e.get("tiling_out_storage_shape") == expected_output]
        infer_exits = [e for e in events if e.get("phase") == "infer" and e.get("event") == "exit"]
        infer = [e for e in events if e.get("phase") == "infer" and e.get("event") == "exit"
                 and e.get("host_ok") is True and e.get("out_shape") == expected_output
                 and (e.get("in_shape"), e.get("perm"), e.get("attr_shape"),
                      e.get("transpose_first")) == expected]
        if not infer_exits:
            infer_unobserved.append(name)
        if name in ("baseline_tf_f32", "rank2_swap_f32") and matches:
            samples.append({"case": name, "infer": infer[0] if infer else None, "tiling": matches[0]})
        if (case["result"] != "PASS_BITWISE" or not matches or (infer_exits and not infer)
                or any("parse_error" in e for e in events)):
            failures.append({"case": name, "result": case["result"], "tiling_matches": len(matches),
                             "infer_success": len(infer), "probe_events": len(events)})
    diagnostic = root / "candidate_boundary" / "results.json"
    boundary = []
    if diagnostic.exists():
        for case in json.loads(diagnostic.read_text(encoding="utf-8")):
            if case["case"] in ("ub_196608_bytes", "crossrank_tf", "infer_tf", "infer_rf"):
                events = records(root / "candidate_boundary" / case["case"] / "run_full.log")
                boundary.append({"case": case["case"], "result": case["result"],
                                 "exit_code": case.get("exit_code"),
                                 "host_exits": [e for e in events if e.get("event") == "exit"]})
    summary = {"case_count": len(cases), "probe_failures": failures, "infer_unobserved": infer_unobserved,
               "samples": samples, "boundary": boundary}
    ub_rejected = any(case["case"] == "ub_196608_bytes" and case["result"] == "FAIL_OR_REJECTED"
                      and any(event.get("phase") == "tiling" and event.get("host_ok") is False
                              and event.get("stage") == "tensor_too_large_for_ub"
                              and event.get("ub_bytes") is not None for event in case["host_exits"])
                      for case in boundary)
    summary["ub_rejection_observed"] = ub_rejected
    (root / "probe_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("C009_PROBE_CHECK", json.dumps(summary, ensure_ascii=False))
    if len(cases) != 39 or failures or len(boundary) != 4 or not ub_rejected:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
