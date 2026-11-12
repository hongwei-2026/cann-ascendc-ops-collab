#!/usr/bin/env python3
"""Check the distinction between tiling metadata and observed infer events."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import c009_probe_report


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reg = self.root / "candidate_regression"
        self.reg.mkdir()
        self.events = {"phase": "tiling", "event": "exit", "host_ok": True, "in_shape": [2, 3],
                       "perm": [1, 0], "attr_shape": [3, 2], "transpose_first": True, "dtype_code": 0,
                       "copy_span": 1, "tiling_out_storage_shape": [3, 2]}
        cases = []
        for i in range(39):
            name = f"case_{i}"
            cases.append({"case": name, "result": "PASS_BITWISE", "in_shape": [2, 3], "perm": [1, 0],
                          "attr_shape": [3, 2], "transpose_first": True})
            folder = self.reg / name
            folder.mkdir()
            (folder / "run.log").write_text("CTD_PROBE " + json.dumps(self.events) + "\n", encoding="utf-8")
        (self.reg / "results.json").write_text(json.dumps(cases), encoding="utf-8")
        boundary = self.root / "candidate_boundary"
        boundary.mkdir()
        names = ["ub_196608_bytes", "crossrank_tf", "infer_tf", "infer_rf"]
        (boundary / "results.json").write_text(json.dumps([
            {"case": name, "result": "FAIL_OR_REJECTED", "exit_code": 2} for name in names
        ]), encoding="utf-8")
        for name in names:
            folder = boundary / name
            folder.mkdir()
            event = {"phase": "tiling", "event": "exit", "host_ok": False,
                     "stage": "tensor_too_large_for_ub", "ub_bytes": 196352}
            (folder / "run_full.log").write_text("CTD_PROBE " + json.dumps(event) + "\n", encoding="utf-8")

    def report(self):
        with contextlib.redirect_stdout(io.StringIO()):
            status = c009_probe_report.main(self.root)
        return status, json.loads((self.root / "probe_check.json").read_text(encoding="utf-8"))

    def test_missing_infer_is_reported_not_claimed_as_success(self):
        status, report = self.report()
        self.assertEqual(status, 0)
        self.assertEqual(len(report["infer_unobserved"]), 39)
        self.assertEqual(report["probe_failures"], [])

    def test_wrong_output_shape_fails(self):
        self.events["tiling_out_storage_shape"] = [2, 3]
        (self.reg / "case_0" / "run.log").write_text("CTD_PROBE " + json.dumps(self.events), encoding="utf-8")
        status, report = self.report()
        self.assertEqual(status, 1)
        self.assertEqual(report["probe_failures"][0]["case"], "case_0")

    def test_observed_infer_mismatch_fails(self):
        event = {"phase": "infer", "event": "exit", "host_ok": True, "out_shape": [2, 3]}
        with (self.reg / "case_0" / "run.log").open("a", encoding="utf-8") as log:
            log.write("CTD_PROBE " + json.dumps(event) + "\n")
        status, report = self.report()
        self.assertEqual(status, 1)
        self.assertNotIn("case_0", report["infer_unobserved"])


if __name__ == "__main__":
    unittest.main()
