#!/usr/bin/env python3
"""CPU-only regression checks for PReluGradUpdate semantics and layout."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "samples"))
import reference  # noqa: E402


def kernel_model(grads, features, weights):
    out_dtype = grads.dtype
    g = grads.astype(np.float32)
    x = features.astype(np.float32)
    w = weights.astype(np.float32)
    flat_g = g.reshape(-1)
    flat_x = x.reshape(-1)
    flat_dx = np.empty_like(flat_g)
    flat_update = np.empty_like(flat_g)
    channel_num = g.shape[1]
    inner_size = int(np.prod(g.shape[2:])) if g.ndim > 2 else 1

    for index in range(flat_g.size):
        channel = (index // inner_size) % channel_num
        weight = w[0] if w.size == 1 else w[channel]
        if flat_x[index] > 0:
            flat_dx[index] = flat_g[index]
            flat_update[index] = 0.0
        else:
            flat_dx[index] = weight * flat_g[index]
            flat_update[index] = flat_x[index] * flat_g[index]

    return (
        flat_dx.reshape(g.shape).astype(out_dtype),
        flat_update.reshape(g.shape).astype(out_dtype),
    )


def check_random_cases():
    rng = np.random.default_rng(20261102)
    shapes = [(4, 6, 8), (2, 3, 5, 7), (1, 1, 17), (3, 257, 1)]
    for dtype in (np.float32, np.float16):
        for shape in shapes:
            grads = rng.uniform(-5, 5, shape).astype(dtype)
            features = rng.uniform(-5, 5, shape).astype(dtype)
            for weights in (
                rng.uniform(0.01, 0.5, (shape[1],)).astype(dtype),
                rng.uniform(0.01, 0.5, (1,)).astype(dtype),
            ):
                expected = reference.impl(grads, features, weights)
                actual = kernel_model(grads, features, weights)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])


def check_select_polarity():
    grads = np.array([[[2.0, 4.0, 6.0, 8.0]]], dtype=np.float32)
    features = np.array([[[1.0, -1.0, 0.0, np.nan]]], dtype=np.float32)
    weights = np.array([0.5], dtype=np.float32)
    dx, update = kernel_model(grads, features, weights)
    np.testing.assert_allclose(dx.reshape(-1), [2.0, 2.0, 3.0, 4.0], equal_nan=True)
    np.testing.assert_allclose(update.reshape(-1), [0.0, -4.0, 0.0, np.nan], equal_nan=True)



def tiling_model(shape, itemsize, ub_size=262144, core_num=64):
    total = int(np.prod(shape))
    channel = shape[1]
    inner = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    sample = channel * inner
    coefficient = 21 if itemsize == 4 else 29
    block_elements = 32 // itemsize
    max_tile = (ub_size * 8 // 10 // coefficient) // block_elements * block_elements
    used_cores = 1
    block_factor = total
    tile_len = block_elements if total == 0 else min(total, max_tile)
    rows_per_tile = 0
    if total > 0 and sample <= max_tile:
        if total <= max_tile:
            rows_per_tile = total // sample
            tile_len = total
        else:
            block_factor = ((total + core_num - 1) // core_num + sample - 1) // sample * sample
            used_cores = min(core_num, (total + block_factor - 1) // block_factor)
            rows_per_tile = min(max(1, max_tile // sample), max(1, block_factor // sample))
            tile_len = rows_per_tile * sample
    elif total > 0:
        block_factor = ((total + core_num - 1) // core_num + block_elements - 1) // block_elements * block_elements
        used_cores = min(core_num, (total + block_factor - 1) // block_factor)
        tile_len = max_tile
    return total, used_cores, block_factor, tile_len, sample, rows_per_tile, max_tile


def check_tiling_invariants():
    shapes = [(4, 6, 8), (4096, 6, 8), (8, 20000, 1), (16, 3, 127, 65)]
    for shape in shapes:
        for itemsize in (2, 4):
            total, cores, block, tile, sample, rows, max_tile = tiling_model(shape, itemsize)
            assert 1 <= cores <= 64
            assert 0 < tile <= max_tile
            assert block > 0
            assert (cores - 1) * block < total <= cores * block
            if rows > 0:
                assert block % sample == 0 or total <= max_tile
                assert tile % sample == 0


def check_weight_fill_alignment():
    float_block_elements = 8
    for inner_size in (1, 3, 5, 7, 8, 35, 8255):
        for global_start in (0, 1, inner_size - 1, inner_size, inner_size + 1):
            current_num = min(257, inner_size * 3 + 5)
            local_offset = 0
            while local_offset < current_num:
                offset_in_channel = (global_start + local_offset) % inner_size
                segment = min(inner_size - offset_in_channel, current_num - local_offset)
                segment_end = local_offset + segment
                fill_offset = local_offset
                while fill_offset < segment_end and fill_offset % float_block_elements != 0:
                    fill_offset += 1
                if fill_offset < segment_end:
                    assert fill_offset % float_block_elements == 0
                local_offset = segment_end

    kernel_source = (ROOT / "op_kernel" / "p_relu_grad_update.h").read_text(encoding="utf-8")
    assert "Duplicate(weightLocal[localOffset]" not in kernel_source
    assert "Duplicate(weightLocal[fillOffset]" in kernel_source


def check_log_signature():
    rng = np.random.default_rng(42)
    rng.uniform(-5, 5, (4, 6, 8))
    features = rng.uniform(-5, 5, (4, 6, 8)).astype(np.float32)
    non_positive = int(np.count_nonzero(~(features > 0)))
    assert non_positive == 95, non_positive


if __name__ == "__main__":
    check_random_cases()
    check_select_polarity()
    check_tiling_invariants()
    check_weight_fill_alignment()
    check_log_signature()
    print("PReluGradUpdate local semantic checks: PASS")
