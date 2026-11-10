#!/usr/bin/env python3
"""CPU oracle for P004 row packing, core boundaries and TF/RF semantics.

This verifies the copy plan against NumPy. It is not a CANN compile or NPU PASS.
"""
from itertools import permutations
from math import gcd, prod

import numpy as np

from local_semantic_check import map_src_index, reference


def copy_span(shape, perm):
    strides = [prod(shape[i + 1:]) for i in range(len(shape))]
    span = 1
    for axis in reversed(perm):
        if shape[axis] == 1:
            continue
        if strides[axis] != span:
            break
        span *= shape[axis]
    return span


def check(shape, perm, attr, tf, width, ub_elements, cores):
    n = prod(shape)
    span = copy_span(shape if tf else attr, perm)
    # Arbitrary byte patterns cover signed integers, NaNs and all storage widths.
    x = np.arange(n * width, dtype=np.uint32).astype(np.uint8).reshape(n, width)
    opaque = np.dtype(f"V{width}")
    expected = reference.impl(x.copy().view(opaque).reshape(shape), list(perm), list(attr), tf)
    expected = np.ascontiguousarray(expected).view(np.uint8).reshape(n, width)
    output = np.full_like(x, 0xA5)
    writes = np.zeros(n, dtype=np.uint8)
    align = 32 // width
    grain = span * (align // gcd(span, align)) if span > 1 else 1
    grains = (n + grain - 1) // grain
    used = min(cores, grains)
    block = ((grains + used - 1) // used) * grain
    padded = ((span + align - 1) // align) * align
    for begin in range(0, n, block):
        end = min(begin + block, n)
        if span == 1:
            for dst in range(begin, end):
                output[dst] = x[map_src_index(dst, shape, perm, attr, tf)]
                writes[dst] += 1
        elif padded > ub_elements:
            for row in range(begin, end, span):
                src = map_src_index(row, shape, perm, attr, tf)
                for start in range(0, span, ub_elements):
                    count = min(span - start, ub_elements)
                    output[row + start:row + start + count] = x[src + start:src + start + count]
                    writes[row + start:row + start + count] += 1
        else:
            rows_per_tile = min(ub_elements // padded, 4095)
            for dst in range(begin, end, rows_per_tile * span):
                rows = min((end - dst) // span, rows_per_tile)
                local = np.full((rows, padded, width), 0xDE, dtype=np.uint8)
                for row in range(rows):
                    src = map_src_index(dst + row * span, shape, perm, attr, tf)
                    local[row, :span] = x[src:src + span]
                output[dst:dst + rows * span] = local[:, :span].reshape(-1, width)
                writes[dst:dst + rows * span] += 1
    assert np.all(writes == 1), (shape, perm, attr, tf, "overlap/gap")
    assert np.array_equal(output, expected), (shape, perm, attr, tf, width, span)
    return span > 1


def main():
    rng = np.random.default_rng(2323004)
    cases = []
    for shape in [(7,), (3, 5), (2, 3, 5), (2, 3, 4, 5), (2, 1, 3, 1)]:
        for perm in permutations(range(len(shape))):
            for tf in (True, False):
                # A same-rank reshape different from the original shape.
                attr = tuple(reversed(shape))
                for width in (1, 2, 4, 8):
                    cases.append((shape, perm, attr, tf, width))
    for rank in range(5, 9):
        for _ in range(24):
            shape = tuple(int(v) for v in rng.integers(1, 4, size=rank))
            perm = tuple(int(v) for v in rng.permutation(rank))
            cases.append((shape, perm, tuple(reversed(shape)), bool(rng.integers(2)), 4))
    fast = 0
    for shape, perm, attr, tf, width in cases:
        for ub, cores in [(1, 1), (32, 3), (128, 40)]:
            fast += check(shape, perm, attr, tf, width, ub, cores)
    # Force multiple 4095-row tiles and an oversized contiguous row.
    check((2, 5000, 2), (1, 0, 2), (5000, 2, 2), True, 1, 200000, 1)
    check((38400,), (0,), (38400,), True, 4, 8192, 40)
    print(f"P004 copy-plan PASS: {len(cases) * 3 + 2} checks; DMA plans={fast}")
    print("CPU oracle only; CANN/NPU verification remains required.")


if __name__ == "__main__":
    main()
