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



def tiling_model(shape, itemsize, ub_size=262144, core_num=64, per_channel_weights=True):
    total = int(np.prod(shape))
    channel = shape[1]
    inner = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    sample = channel * inner
    coefficient = 25 if itemsize == 4 else 33
    block_elements = 32 // itemsize
    cmp_align_elements = 64
    max_tile = (ub_size * 8 // 10 // coefficient) // cmp_align_elements * cmp_align_elements
    used_cores = 1
    block_factor = total
    tile_len = block_elements if total == 0 else min(total, max_tile)
    rows_per_tile = 0
    if total > 0:
        block_factor = ((total + core_num - 1) // core_num + block_elements - 1) // block_elements * block_elements
        used_cores = min(core_num, (total + block_factor - 1) // block_factor)
        tile_len = min(block_factor, max_tile)
        if per_channel_weights and inner > 0:
            tile_len = min(tile_len, inner)
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


def check_channel_tile_boundaries():
    shapes = [(4, 6, 8), (2, 3, 5, 7), (8, 20000, 1), (16, 3, 127, 65)]
    for shape in shapes:
        inner_size = int(np.prod(shape[2:])) if len(shape) > 2 else 1
        for itemsize in (2, 4):
            total, cores, block, tile, _, _, _ = tiling_model(shape, itemsize)
            for core_index in range(cores):
                block_start = core_index * block
                block_length = min(block, total - block_start)
                progress = 0
                while progress < block_length:
                    global_start = block_start + progress
                    current_num = min(tile, block_length - progress)
                    current_num = min(current_num, inner_size - global_start % inner_size)
                    assert current_num > 0
                    first_channel = global_start // inner_size
                    last_channel = (global_start + current_num - 1) // inner_size
                    assert first_channel == last_channel
                    progress += current_num

    kernel_source = (ROOT / "op_kernel" / "p_relu_grad_update.h").read_text(encoding="utf-8")
    assert "weightFloatBuf_" not in kernel_source
    assert "Duplicate(weightLocal" not in kernel_source
    assert "Muls(negLocal, gradsFloat, weightValue, currentNum)" in kernel_source
    assert "progress += currentNum" in kernel_source
    assert "for (int64_t progress = 0; progress < blockLength_; progress += tileLen_)" not in kernel_source
    # P102: aligned contiguous copy gate must stay dual-path (DataCopy + DataCopyPad).
    assert "globalOffset % blockElements == 0 && currentNum % blockElements == 0" in kernel_source
    assert "DataCopy(gradsLocal, gradsGm_[globalOffset]" in kernel_source
    assert "DataCopyPad(gradsLocal, gradsGm_[globalOffset]" in kernel_source


def check_full_core_tiling():
    total, cores, block, tile, sample, rows, _ = tiling_model((4, 6, 8), 4)
    assert total == 192
    assert cores == 24
    assert block == 8
    assert tile == 8
    assert sample == 48
    assert rows == 0

    _, cores, block, tile, _, _, _ = tiling_model((4, 6, 8), 2)
    assert cores == 12
    assert block == 16
    assert tile == 8  # capped to innerSize when weights are [C]


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
    check_channel_tile_boundaries()
    check_full_core_tiling()
    check_log_signature()
    print("PReluGradUpdate local semantic checks: PASS")
