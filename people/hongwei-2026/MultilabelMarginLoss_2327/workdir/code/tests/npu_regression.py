#!/usr/bin/env python3
"""Reference-based NPU regression for zero/small workspace and row reductions.

After build.sh -e, run python3 tests/npu_regression.py. --generate-only prepares
inputs/golden on CPU without claiming that the kernel has passed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'samples'))
import reference

SHAPES = [(4, 6, 6), (8, 32, 16), (8, 32, 17), (9, 17, 11),
          (100, 6, 6), (1, 257, 11), (3, 16385, 11), (513, 6, 9)]
DTYPES = {'float32': np.float32, 'float16': np.float16, 'bfloat16': ml_dtypes.bfloat16}


def targets(rng, n, c, k):
    values = rng.integers(0, c + 2, size=(n, k), dtype=np.int32)
    for i in range(n):
        if i % 6 == 0:
            values[i, :min(k, 3)] = 0  # duplicate counts
        elif i % 6 == 1:
            values[i, 0] = -2  # arbitrary negative, not only -1
        elif i % 6 == 2:
            values[i, :] = c + 7  # out-of-range positives do not terminate
        elif i % 6 == 3:
            values[i, k // 2] = -1  # labels after this must be ignored
        elif i % 6 == 4:
            values[i, :] = np.arange(k) % c  # no padding, includes all if k>=c
        if i % 6 == 0 and k > 3:
            values[i, 3] = -1
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--directory', type=Path, default=ROOT / 'samples' / 'build-m009-regression')
    parser.add_argument('--binary', type=Path, default=ROOT / 'examples' / 'build' / 'sample')
    parser.add_argument('--device', default=os.environ.get('DEVICE_ID', '0'))
    args = parser.parse_args()
    if not args.generate_only and not args.binary.is_file():
        parser.error('Build the NPU example first with DEVICE_ID=0 bash build.sh -j4 -e')
    rng = np.random.default_rng(2327009)
    env = os.environ.copy()
    vendor = ROOT / 'build/packages/vendors/multilabel_margin_loss_custom'
    env['ASCEND_CUSTOM_OPP_PATH'] = str(vendor) + (':' + env['ASCEND_CUSTOM_OPP_PATH']
        if env.get('ASCEND_CUSTOM_OPP_PATH') else '')
    env['LD_LIBRARY_PATH'] = str(vendor / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    results = []
    for n, c, k in SHAPES:
        raw_x = rng.uniform(-5, 5, (n, c))
        target = targets(rng, n, c, k)
        for dtype_name, dtype in DTYPES.items():
            x = raw_x.astype(dtype)
            for reduction in ('none', 'mean', 'sum'):
                case_name = f'n{n}_c{c}_k{k}_{dtype_name}_{reduction}'
                folder = args.directory / case_name
                for sub in ('input', 'golden', 'output'):
                    (folder / sub).mkdir(parents=True, exist_ok=True)
                (folder / 'input/x.bin').write_bytes(x.tobytes())
                (folder / 'input/target.bin').write_bytes(target.tobytes())
                expected = reference.impl(x, target, reduction=reduction)
                for name, array in zip(('y', 'is_target'), expected):
                    (folder / f'golden/{name}.bin').write_bytes(array.tobytes())
                record = {'case': case_name, 'result': 'GENERATED_ONLY'}
                if not args.generate_only:
                    # Remove stale outputs, and validate both exact byte lengths.
                    for name in ('y', 'is_target'):
                        (folder / f'output/{name}.bin').unlink(missing_ok=True)
                    proc = subprocess.run([str(args.binary.resolve()), str(folder.resolve()), args.device,
                        str(n), str(c), str(k), dtype_name, reduction], env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
                    (folder / 'run.log').write_text(proc.stdout, encoding='utf-8')
                    ok = proc.returncode == 0
                    for name, golden in zip(('y', 'is_target'), expected):
                        path = folder / f'output/{name}.bin'
                        if not path.is_file() or path.stat().st_size != golden.nbytes:
                            ok = False
                            continue
                        actual = np.frombuffer(path.read_bytes(), dtype=golden.dtype).reshape(golden.shape)
                        if name == 'is_target':
                            ok &= bool(np.array_equal(actual, golden))
                        else:
                            ok &= bool(np.allclose(actual.astype(np.float64), golden.astype(np.float64),
                                rtol=1e-4, atol=1e-4, equal_nan=True))
                    record['result'] = 'PASS' if ok else 'FAIL'
                    print(record['result'], case_name, flush=True)
                results.append(record)
    args.directory.mkdir(parents=True, exist_ok=True)
    (args.directory / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    if args.generate_only:
        print(f'Generated {len(results)} reference cases; NPU NOT RUN.')
    else:
        failed = sum(r['result'] != 'PASS' for r in results)
        print(f'NPU regression: {len(results)-failed}/{len(results)} PASS')
        raise SystemExit(bool(failed))


if __name__ == '__main__':
    main()
