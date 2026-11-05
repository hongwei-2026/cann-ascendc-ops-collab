#!/usr/bin/env python3
"""J001 CPU schedule model; does not compile or execute the AscendC kernel."""
import hashlib
import json
from pathlib import Path
import zipfile

import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT.parents[1]


def chunks(n, c, s, wlen, cores, itemsize, candidate):
    total = n * c * s
    align = 32 // itemsize
    block = ((total + cores - 1) // cores + align - 1) // align * align
    # Simulated 256 KiB UB, with the same 8 KiB reserve and 80% budget.
    ub = 256 * 1024 - 8192
    coefficient = 25 if itemsize == 4 else 33
    max_tile = (ub * 8 // 10 // coefficient) // 64 * 64
    channel = wlen > 1 if candidate else wlen == c
    tile = min(block, max_tile, s) if channel else min(block, max_tile)
    assert ((tile + 63) // 64 * 64) * coefficient <= ub * 8 // 10
    for begin in range(0, total, block):
        end = min(begin + block, total)
        while begin < end:
            count = min(tile, end - begin)
            if channel:
                count = min(count, s - begin % s)
            yield begin, count, (begin // s % c if channel else 0)
            begin += count


def check_case(shape, wlen, cores, dtype, rng):
    n, c, s = shape
    total = n * c * s
    x = rng.uniform(-2, 2, total).astype(dtype).astype(np.float32)
    g = rng.uniform(-2, 2, total).astype(dtype).astype(np.float32)
    w = rng.uniform(-2, 2, wlen).astype(dtype).astype(np.float32)
    x[::17] = 0
    index = np.zeros(total, dtype=np.int64) if wlen == 1 else np.arange(total) // s % c
    expected_dx = np.where(x > 0, g, g * w[index]).astype(dtype)
    expected_up = np.where(x > 0, np.float32(0), x * g).astype(dtype)
    counts = []
    for candidate in (False, True):
        seen = np.zeros(total, dtype=np.uint8)
        dx, up = np.empty(total, dtype=dtype), np.empty(total, dtype=dtype)
        count = 0
        for begin, size, wi in chunks(n, c, s, wlen, cores, np.dtype(dtype).itemsize, candidate):
            end = begin + size
            assert np.all(index[begin:end] == wi)
            seen[begin:end] += 1
            dx[begin:end] = np.where(x[begin:end] > 0, g[begin:end], g[begin:end] * w[wi]).astype(dtype)
            up[begin:end] = np.where(x[begin:end] > 0, np.float32(0), x[begin:end] * g[begin:end]).astype(dtype)
            count += 1
        assert np.all(seen == 1)
        np.testing.assert_array_equal(dx, expected_dx)
        np.testing.assert_array_equal(up, expected_up)
        counts.append(count)
    if c != 1:
        assert counts[0] == counts[1]
    return counts


def main():
    host = (ROOT / 'op_host/p_relu_grad_update_tiling.cpp').read_text(encoding='utf-8')
    assert 'if (weightLen > 1 && innerSize > 0)' in host
    assert 'tiling->weightsPerChannel = weightLen > 1 ? 1 : 0;' in host
    assert 'weightLen == channelNum' not in host
    kernel = (ROOT / 'op_kernel/p_relu_grad_update.h').read_bytes()
    with zipfile.ZipFile(TASK / 'baseline/P103-code.zip') as baseline:
        assert kernel == baseline.read('code/op_kernel/p_relu_grad_update.h')
    rng = np.random.default_rng(2329)
    cases = elements = old_tiles = new_tiles = 0
    for n in (1, 2, 7):
        for c in (1, 3, 17):
            for s in (1, 3, 7, 64, 65, 257):
                for wlen in sorted({1, c}):
                    for cores in (1, 8, 48):
                        for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
                            old, new = check_case((n, c, s), wlen, cores, dtype, rng)
                            cases += 1
                            elements += n * c * s
                            old_tiles += old
                            new_tiles += new
    # Larger C=1 inputs force multiple tiles, tails, and cross-batch execution.
    for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
        for shape in ((9001, 1, 1), (533, 1, 17)):
            old, new = check_case(shape, 1, 1, dtype, rng)
            cases += 1
            elements += int(np.prod(shape))
            old_tiles += old
            new_tiles += new
    counts = {}
    for dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
        args = (160000, 1, 1, 1, 48, np.dtype(dtype).itemsize)
        counts[str(np.dtype(dtype))] = [sum(1 for _ in chunks(*args, candidate)) for candidate in (False, True)]
    print(json.dumps({
        'scope': 'CPU model only; CANN Host UT and NPU execution NOT run',
        'result': 'PASS', 'seed': 2329, 'cases': cases,
        'elements_per_model': elements, 'baseline_tiles': old_tiles, 'candidate_tiles': new_tiles,
        'modeled_160000_1_1_48cores_tile_counts': counts,
        'kernel_identical_to_P103': True,
        'kernel_sha256': hashlib.sha256(kernel).hexdigest(),
        'host_sha256': hashlib.sha256((ROOT / 'op_host/p_relu_grad_update_tiling.cpp').read_bytes()).hexdigest(),
        'limitations': 'Finite-value software model, not device dtype behavior or performance. Public Judge shapes unknown.'
    }, indent=2))


if __name__ == '__main__':
    main()
