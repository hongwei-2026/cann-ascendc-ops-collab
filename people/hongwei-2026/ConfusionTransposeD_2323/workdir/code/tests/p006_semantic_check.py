#!/usr/bin/env python3
"""Check DMA addresses, padding and output bytes against an independent NumPy transpose."""
from itertools import permutations
from math import prod

import numpy as np

from local_semantic_check import map_src_index, reference
from p004_semantic_check import copy_span
from p006_regression import STRIDED_CASES


def check(shape, perm, attr, tf, width, cores, ub_bytes=192 * 1024, slot_limit=4095):
    dims = shape if tf else attr
    if copy_span(dims, perm) != 1:
        return False
    axis = next((p for p in reversed(perm) if dims[p] > 1), None)
    if axis is None:
        return False
    span = dims[axis]
    step = prod(dims[axis + 1:])
    assert step > 1 and (step - 1) * width <= 0xFFFFFFFF
    n = prod(shape)
    assert n * width + 8192 < ub_bytes
    align = 32 // width
    grains = (n + align - 1) // align
    used = min(cores, grains)
    block = ((grains + used - 1) // used) * align
    slots = min(block, (ub_bytes - 8192) // 32, slot_limit) // align * align
    assert slots >= align and slots <= 4095 and slots * 32 + 8192 <= ub_bytes
    rng = np.random.default_rng(n + width)
    x = rng.integers(0, 256, size=(n, width), dtype=np.uint8)
    oracle = reference.impl(x.copy().view(f"V{width}").reshape(shape), list(perm), list(attr), tf)
    expected = np.ascontiguousarray(oracle).view(np.uint8).reshape(n, width)
    output = np.full_like(x, 0xCD)
    writes = np.zeros(n, dtype=np.uint8)
    for begin in range(0, n, block):
        assert begin * width % 32 == 0
        end = min(begin + block, n)
        for dst in range(begin, end, slots):
            count = min(end - dst, slots)
            assert dst * width % 32 == 0
            local = np.full((slots, 32), 0xA5, dtype=np.uint8)
            offset = 0
            while offset < count:
                chunk = min(count - offset, span - ((dst + offset) % span))
                src = map_src_index(dst + offset, shape, perm, attr, tf)
                indexes = src + np.arange(chunk) * step
                assert indexes.min() >= 0 and indexes.max() < n
                local[offset:offset + chunk, :width] = x[indexes]
                offset += chunk
            output[dst:dst + count] = local[:count, :width]
            writes[dst:dst + count] += 1
    assert np.all(writes == 1), (shape, perm, attr, tf, "overlap/gap")
    assert np.array_equal(output, expected), (shape, perm, attr, tf, width, span, step)
    return True


def main():
    checks = 0
    for shape in [(3, 5), (2, 3, 5), (2, 3, 4, 5), (3, 5, 1), (2, 1, 3, 1)]:
        for perm in permutations(range(len(shape))):
            for tf in (True, False):
                for width in (1, 2, 4, 8):
                    for cores in (1, 3, 40):
                        checks += check(shape, perm, tuple(reversed(shape)), tf, width, cores)
    rng = np.random.default_rng(2323006)
    for rank in range(5, 9):
        for _ in range(24):
            shape = tuple(int(v) for v in rng.integers(1, 4, size=rank))
            perm = tuple(int(v) for v in rng.permutation(rank))
            checks += check(shape, perm, tuple(reversed(shape)), bool(rng.integers(2)), 4, 40)
    widths = dict(float32=4, float16=2, int8=1, int64=8, bfloat16=2)
    for _, shape, perm, attr, tf, dtype in STRIDED_CASES:
        checks += check(shape, perm, attr, tf, widths[dtype], 40)
    # Force multi-tile execution and block-count ceiling; the cloud uses real 40-core limits.
    checks += check((128, 256), (1, 0), (256, 128), True, 4, 1)
    checks += check((7, 129), (1, 0), (129, 7), True, 1, 3, slot_limit=64)
    checks += check((128, 64), (2, 0, 1), (16, 16, 32), False, 2, 3)
    print(f"P006 strided DMA address/byte oracle: {checks} plans PASS")
    print("CPU model only; srcStride hardware behavior and performance require CANN/NPU verification.")


if __name__ == "__main__":
    main()
