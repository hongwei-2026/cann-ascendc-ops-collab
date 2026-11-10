#!/usr/bin/env python3
"""P005: probe existing P001/P004 boundaries without changing Host/Kernel.

Reuse C004 builds. Run each case in a separate process; label host wall time
explicitly. Probe failures are evidence, never a hidden-case diagnosis or PASS.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from npu_regression import CASES, P004_CASES, DTYPE_MAP, reference, write_meta


def cases():
    result = [(c, "control_22") for c in CASES + P004_CASES]
    for n in (1, 2, 3, 7, 9):
        for dtype in ("int8", "float16", "float32"):
            result.append(((f"tiny_{n}_{dtype}", (n,), (0,), (n,), True, dtype), "tiny_alignment"))
    for dtype in ("uint8", "int16", "uint16", "uint32", "uint64"):
        result.append(((f"storage_{dtype}", (3, 5, 7, 3), (0, 2, 1, 3),
                        (3, 7, 5, 3), True, dtype), "uncovered_storage_dtype"))
    for n in (32768, 40960, 47104, 49152, 65536):
        result.append(((f"ub_{n * 4}_bytes", (2, 2, n // 4), (1, 0, 2),
                        (2, 2, n // 4), True, "float32"), "UB_boundary_probe"))
    result += [
        ((("crossrank_tf", (2, 3, 4, 5), (0, 2, 1, 3), (6, 20), True, "float32")), "crossrank_probe"),
        ((("crossrank_rf", (6, 20), (0, 2, 1, 3), (2, 4, 3, 5), False, "float32")), "crossrank_probe"),
        ((("infer_tf", (2, 3, 4, 5), (0, 2, 1, 3), (2, 4, 3, -1), True, "float32")), "inferred_dim_probe"),
        ((("infer_rf", (2, 3, 4, 5), (0, 2, 1, 3), (2, 4, 3, -1), False, "float32")), "inferred_dim_probe"),
        ((("empty_tensor", (0, 3), (1, 0), (3, 0), True, "float32")), "empty_probe"),
    ]
    return result


def make_diagnostic_source(original):
    """Only shared example driver changes; the tested Host/Kernel remain intact."""
    replacements = [
        ('#include <sstream>', '#include <sstream>\n#include <chrono>'),
        ('if (meta.perm.size() != meta.outShape.size()) {',
         'if (meta.perm.size() != (meta.transposeFirst ? meta.inShape.size() : meta.outShape.size())) {'),
        ('if (total <= 0 || Numel(meta.outShape) != total) {',
         'if (total < 0 || (std::count(meta.outShape.begin(), meta.outShape.end(), -1) == 0 && Numel(meta.outShape) != total)) {'),
        ('std::vector<int64_t> finalShape = meta.outShape;', '''std::vector<int64_t> resolvedShape = meta.outShape;
        if (std::count(resolvedShape.begin(), resolvedShape.end(), -1) == 1) {
            int64_t known = 1;
            for (auto dim : resolvedShape) if (dim != -1) known *= dim;
            if (known <= 0 || total % known != 0) throw std::runtime_error("bad inferred shape");
            for (auto& dim : resolvedShape) if (dim == -1) dim = total / known;
        }
        std::vector<int64_t> finalShape = resolvedShape;'''),
        ('finalShape[i] = meta.outShape[static_cast<size_t>(meta.perm[i])];',
         'finalShape[i] = resolvedShape[static_cast<size_t>(meta.perm[i])];'),
        ('check(aclInit(nullptr));', '''using Clock = std::chrono::steady_clock;
        auto stamp = Clock::now();
        auto mark = [&](const char* phase) {
            auto now = Clock::now();
            std::cout << "DIAG phase=" << phase << " host_wall_us="
                << std::chrono::duration<double, std::micro>(now - stamp).count() << std::endl;
            stamp = now;
        };
        std::cout << "DIAG begin=aclInit" << std::endl;
        check(aclInit(nullptr));
        mark("aclInit");'''),
        ('uint64_t size = 0;', 'mark("input_output_setup");\n            uint64_t size = 0;'),
        ('check(aclnnConfusionTransposeDGetWorkspaceSize(',
         'std::cout << "DIAG begin=GetWorkspaceSize" << std::endl;\n            check(aclnnConfusionTransposeDGetWorkspaceSize('),
        ('std::cout << "workspace_bytes="', 'mark("GetWorkspaceSize");\n            std::cout << "workspace_bytes="'),
        ('check(aclnnConfusionTransposeD(workspace, size, executor, stream));',
         'mark("pre_launch_setup");\n                std::cout << "DIAG begin=launch" << std::endl;\n                check(aclnnConfusionTransposeD(workspace, size, executor, stream));\n                mark("launch_api");'),
        ('check(aclrtSynchronizeStream(stream));',
         'std::cout << "DIAG begin=synchronize" << std::endl;\n                check(aclrtSynchronizeStream(stream));\n                mark("synchronize");'),
    ]
    for old, new in replacements:
        expected = 2 if old == 'check(aclnnConfusionTransposeDGetWorkspaceSize(' else 1
        if original.count(old) != expected:
            raise ValueError(f"Source anchor changed: {old!r}")
        original = original.replace(old, new)
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    import os
    env = os.environ.copy()
    vendor = args.code_root / "build/packages/vendors/confusion_transpose_d_custom"
    env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor.resolve())
    env["LD_LIBRARY_PATH"] = str(vendor.resolve() / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CTD_PERF_WARMUP"] = "0"
    env["CTD_PERF_REPEAT"] = "1"
    env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
    env["ASCEND_GLOBAL_LOG_LEVEL"] = "0"
    binary = args.directory / "diagnostic_sample"
    if not args.generate_only:
        source = args.directory / "diagnostic_sample.cpp"
        source.write_text(make_diagnostic_source((args.code_root / "examples/test_aclnn_confusion_transpose_d.cpp").read_text(encoding="utf-8")), encoding="utf-8")
        toolkit = Path(os.environ["ASCEND_HOME_PATH"])
        command = ["c++", "-std=c++17", str(source), f"-I{args.code_root / 'examples'}",
                   f"-I{vendor / 'op_api/include'}", f"-I{toolkit / 'include'}",
                   f"-L{vendor / 'op_api/lib'}", "-lcust_opapi", f"-L{toolkit / 'lib64'}",
                   "-lascendcl", "-lnnopbase", f"-Wl,-rpath,{vendor / 'op_api/lib'}",
                   "-o", str(binary)]
        built = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (args.directory / "driver_build.log").write_bytes(built.stdout)
        (args.directory / "driver_command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        if built.returncode:
            print("DIAGNOSTIC_DRIVER_BUILD_FAILED", flush=True)
            return 2
    dtypes = dict(DTYPE_MAP, uint8=np.uint8, int16=np.int16, uint16=np.uint16, uint32=np.uint32, uint64=np.uint64)
    rng = np.random.default_rng(2323005)
    results = []
    stop_device = False
    for case, category in cases():
        name, shape, perm, attr, tf, dtype_name = case
        record = dict(case=name, category=category, version=args.version, in_shape=shape,
                      perm=perm, attr_shape=attr, transpose_first=tf, dtype=dtype_name)
        folder = args.directory / name
        for sub in ("input", "golden", "output"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        if dtypes.get(dtype_name) is None:
            record["result"] = "SKIP_NO_DTYPE"
        elif stop_device:
            record["result"] = "SKIP_AFTER_DEVICE_ERROR"
        else:
            dtype = np.dtype(dtypes[dtype_name])
            # Arbitrary bits verify that float and integer payloads survive unchanged.
            raw = rng.integers(0, 256, size=int(np.prod(shape)) * dtype.itemsize, dtype=np.uint8)
            x = raw.view(dtype).reshape(shape)
            y = np.ascontiguousarray(reference.impl(x, list(perm), list(attr), tf))
            (folder / "input/x.bin").write_bytes(x.tobytes())
            golden = y.tobytes()
            (folder / "golden/y.bin").write_bytes(golden)
            write_meta(folder, dtype_name, tf, shape, perm, attr)
            record.update(input_bytes=x.nbytes, golden_sha256=hashlib.sha256(golden).hexdigest())
            if args.generate_only:
                record["result"] = "GENERATED_ONLY"
            else:
                started = time.monotonic()
                try:
                    proc = subprocess.run([str(binary.resolve()), str(folder.resolve()), args.device], env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
                    (folder / "run_full.log").write_bytes(proc.stdout)
                    record["exit_code"] = proc.returncode
                    output = folder / "output/y.bin"
                    record["result"] = ("PASS_BITWISE" if proc.returncode == 0 and output.is_file()
                                        and output.read_bytes() == golden else "FAIL_OR_REJECTED")
                    record["host_phases"] = [line for line in proc.stdout.decode(errors="replace").splitlines()
                                             if line.startswith("DIAG ")]
                    if proc.returncode < 0:
                        stop_device = True
                except subprocess.TimeoutExpired as error:
                    (folder / "run_full.log").write_bytes((error.stdout or b"") + b"\nTIMEOUT after 30s\n")
                    record["result"] = "TIMEOUT"
                    stop_device = True
                record["process_wall_seconds"] = time.monotonic() - started
                record["timing_scope"] = "host/process wall time, not kernel task time"
        results.append(record)
        (args.directory / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(args.version, name, record["result"], flush=True)
    counts = dict(collections.Counter(r["result"] for r in results))
    (args.directory / "diagnostic_status.json").write_text(json.dumps(
        {"probe_cases": len(results), "counts": counts, "meaning": "Diagnostic completed; failures remain failures, not kernel acceptance."}, indent=2) + "\n", encoding="utf-8")
    print("DIAGNOSTIC_COUNTS", json.dumps(counts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
