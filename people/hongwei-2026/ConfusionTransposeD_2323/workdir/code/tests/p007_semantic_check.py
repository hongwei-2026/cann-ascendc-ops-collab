#!/usr/bin/env python3
"""CPU oracle for P007 regular-stride row batching; not a hardware check."""
from math import gcd, prod

import numpy as np

from local_semantic_check import map_src_index, reference
from npu_regression import CASES as REGRESSION_CASES, P004_CASES
from p004_semantic_check import copy_span


WIDTHS = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int32": 4, "int64": 8}


def run_case(shape, perm, attr, tf, width, ub_elements, cores):
    n = prod(shape)
    opaque = np.dtype(f"V{width}")
    seed = 2323007 + n * width + len(shape) * 19 + sum(perm)
    source = np.random.default_rng(seed).integers(0, 256, size=(n, width), dtype=np.uint8)
    source_items = source.copy().view(opaque).reshape(shape)
    expected = reference.impl(source_items, list(perm), list(attr), tf)
    expected = np.ascontiguousarray(expected).view(np.uint8).reshape(n, width)
    output = np.full_like(source, 0xA5)
    writes = np.zeros(n, dtype=np.uint8)
    input_calls = 0

    span = 1 if tf and len(shape) != len(attr) else copy_span(shape if tf else attr, perm)
    if span == 1:
        for dst in range(n):
            src = map_src_index(dst, shape, perm, attr, tf)
            output[dst] = source[src]
            writes[dst] += 1
            input_calls += 1
    else:
        align = 32 // width
        grain = span * (align // gcd(span, align))
        grains = (n + grain - 1) // grain
        used = min(cores, grains)
        block = ((grains + used - 1) // used) * grain
        padded = ((span + align - 1) // align) * align
        for begin in range(0, n, block):
            end = min(begin + block, n)
            if padded > ub_elements:
                for row_dst in range(begin, end, span):
                    src = map_src_index(row_dst, shape, perm, attr, tf)
                    for start in range(0, span, ub_elements):
                        count = min(span - start, ub_elements)
                        output[row_dst + start:row_dst + start + count] = source[src + start:src + start + count]
                        writes[row_dst + start:row_dst + start + count] += 1
                        input_calls += 1
                continue

            rows_per_tile = min(ub_elements // padded, 4095)
            tile = begin
            while tile < end:
                rows = min((end - tile) // span, rows_per_tile)
                dst = tile
                while dst < tile + rows * span:
                    max_rows = min((tile + rows * span - dst) // span, rows_per_tile)
                    row_bytes = span * width
                    src = map_src_index(dst, shape, perm, attr, tf)
                    batch_rows = 1
                    can_batch = row_bytes % 32 == 0 and max_rows > 1
                    if can_batch:
                        src_step = map_src_index(dst + span, shape, perm, attr, tf) - src
                        can_batch = src_step >= span and (src_step - span) * width <= 0xFFFFFFFF
                        if can_batch:
                            batch_rows = 2
                            for row in range(2, max_rows):
                                if map_src_index(dst + row * span, shape, perm, attr, tf) != src + row * src_step:
                                    break
                                batch_rows += 1
                    if can_batch and batch_rows > 1:
                        for row in range(batch_rows):
                            row_dst = dst + row * span
                            row_src = src + row * src_step
                            output[row_dst:row_dst + span] = source[row_src:row_src + span]
                            writes[row_dst:row_dst + span] += 1
                        input_calls += 1
                        dst += batch_rows * span
                        continue

                    for row in range(max_rows):
                        row_dst = dst + row * span
                        row_src = map_src_index(row_dst, shape, perm, attr, tf)
                        output[row_dst:row_dst + span] = source[row_src:row_src + span]
                        writes[row_dst:row_dst + span] += 1
                        input_calls += 1
                    dst += max_rows * span
                tile += rows * span

    assert np.all(writes == 1), (shape, perm, attr, tf, "overlap/gap")
    assert np.array_equal(output, expected), (shape, perm, attr, tf, width, span)
    return input_calls


def main():
    checks = 0
    target_counts = None
    for name, shape, perm, attr, tf, dtype in REGRESSION_CASES + P004_CASES:
        width = WIDTHS[dtype]
        for ub, cores in ((64, 1), (512, 8), (32768, 40)):
            calls = run_case(shape, perm, attr, tf, width, ub, cores)
            checks += 1
            if name == "dma_rows128_f32" and ub == 32768 and cores == 40:
                span = copy_span(shape, perm)
                rows = prod(shape) // span
                target_counts = rows, calls

    rng = np.random.default_rng(2323007)
    for rank in range(2, 7):
        for _ in range(20):
            shape = tuple(int(v) for v in rng.integers(2, 5, size=rank))
            perm = tuple(int(v) for v in rng.permutation(rank))
            for tf in (True, False):
                run_case(shape, perm, tuple(reversed(shape)), tf, 4, 256, 8)
                checks += 1

    assert target_counts is not None and target_counts[1] < target_counts[0], target_counts
    print(f"P007 row-batch semantic plan PASS: {checks} checks; dma_rows128 input calls {target_counts[0]} -> {target_counts[1]}")
    print("CPU oracle only; CANN DataCopyPad srcStride behavior and timing require CodeArts verification.")


if __name__ == "__main__":
    main()
