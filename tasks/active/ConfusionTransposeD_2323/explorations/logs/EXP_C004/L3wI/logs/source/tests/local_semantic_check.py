#!/usr/bin/env python3
"""CPU-only semantic checks for ConfusionTransposeD reference + index remap."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "samples"))
import reference  # noqa: E402

CASES = [
    # name, in_shape, perm, out_shape, transpose_first
    ("baseline_tf", (2, 3, 4, 5), (0, 2, 1, 3), (2, 4, 3, 5), True),
    ("baseline_rf", (2, 3, 4, 5), (0, 2, 1, 3), (2, 4, 3, 5), False),
    ("rank2_swap", (8, 16), (1, 0), (16, 8), True),
    ("rank3_rot", (4, 5, 6), (2, 0, 1), (6, 4, 5), True),
    ("rank3_rf", (4, 5, 6), (2, 0, 1), (6, 4, 5), False),
    ("reshape_same_rank_tf", (2, 3, 4, 5), (0, 2, 1, 3), (4, 3, 2, 5), True),
    ("reshape_same_rank_rf", (2, 3, 4, 5), (0, 2, 1, 3), (4, 5, 3, 2), False),
    ("larger_4d", (4, 8, 8, 8), (0, 2, 1, 3), (4, 8, 8, 8), True),
]


def map_src_index(dst_index, in_shape, perm, out_shape, transpose_first):
    """Mirror kernel MapSrcIndex (flat dst → flat src)."""
    rank = len(out_shape)
    rank_in = len(in_shape)
    dim_in = list(in_shape)
    dim_shape = list(out_shape) if not transpose_first else list(out_shape)
    # For TF=false attr shape is reshape mid; for TF=true attr shape is final out.
    # Kernel stores dimShape_ from attr `shape` always.
    attr_shape = list(out_shape)
    if transpose_first:
        t_shape = [dim_in[p] for p in perm]
        # decode dst with t_shape strides
        strides_t = [0] * rank
        strides_t[-1] = 1
        for i in range(rank - 2, -1, -1):
            strides_t[i] = strides_t[i + 1] * t_shape[i + 1]
        rem = dst_index
        src = 0
        strides_in = [0] * rank_in
        strides_in[-1] = 1
        for i in range(rank_in - 2, -1, -1):
            strides_in[i] = strides_in[i + 1] * dim_in[i + 1]
        for i in range(rank):
            coord = rem // strides_t[i]
            rem = rem % strides_t[i]
            src += coord * strides_in[perm[i]]
        return src
    # reshape first: attr_shape is mid; final out = permute(mid)
    mid = list(attr_shape)
    out = [mid[p] for p in perm]
    strides_out = [0] * rank
    strides_out[-1] = 1
    for i in range(rank - 2, -1, -1):
        strides_out[i] = strides_out[i + 1] * out[i + 1]
    strides_shape = [0] * rank
    strides_shape[-1] = 1
    for i in range(rank - 2, -1, -1):
        strides_shape[i] = strides_shape[i + 1] * mid[i + 1]
    rem = dst_index
    mid_coord = [0] * rank
    for i in range(rank):
        coord = rem // strides_out[i]
        rem = rem % strides_out[i]
        mid_coord[perm[i]] = coord
    src = 0
    for i in range(rank):
        src += mid_coord[i] * strides_shape[i]
    return src


def check_case(name, in_shape, perm, out_shape, transpose_first, dtype=np.float32):
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    x = rng.uniform(-5, 5, in_shape).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        x = rng.integers(-8, 8, in_shape, dtype=dtype)
    y = reference.impl(x, list(perm), list(out_shape), transpose_first)
    # For TF=false, reference out shape is permute(reshape(x, shape))
    # Our case table uses out_shape as the attr `shape` for both modes when TF=true;
    # when TF=false, out_shape in CASES is the attr shape (mid), and final y shape differs.
    attr_shape = list(out_shape)
    if transpose_first:
        assert list(y.shape) == list(attr_shape), (name, y.shape, attr_shape)
        final_shape = attr_shape
        # gather via index map on flattened final
        flat_y = y.reshape(-1)
        rebuilt = np.empty_like(flat_y)
        for dst in range(flat_y.size):
            src = map_src_index(dst, in_shape, perm, attr_shape, True)
            rebuilt[dst] = x.reshape(-1)[src]
        if not np.array_equal(rebuilt, flat_y):
            raise AssertionError(f"{name}: index remap mismatch TF=true")
    else:
        # attr shape = mid; final = permute(mid)
        mid = attr_shape
        final_shape = [mid[p] for p in perm]
        assert list(y.shape) == final_shape, (name, y.shape, final_shape)
        flat_y = y.reshape(-1)
        rebuilt = np.empty_like(flat_y)
        for dst in range(flat_y.size):
            src = map_src_index(dst, in_shape, perm, attr_shape, False)
            rebuilt[dst] = x.reshape(-1)[src]
        if not np.array_equal(rebuilt, flat_y):
            raise AssertionError(f"{name}: index remap mismatch TF=false")
    print("PASS", name, "in", in_shape, "attr", attr_shape, "out", list(y.shape), "tf", transpose_first)


def main():
    reference.impl(np.zeros((2, 3, 4, 5), np.float32), [0, 2, 1, 3], [2, 4, 3, 5], True)
    for case in CASES:
        check_case(*case)
    # dtype smoke on baseline via reference only
    for dtype in (np.float16, np.int8, np.int32):
        x = np.arange(2 * 3 * 4 * 5, dtype=dtype).reshape(2, 3, 4, 5)
        y = reference.impl(x, [0, 2, 1, 3], [2, 4, 3, 5], True)
        assert y.dtype == dtype and y.shape == (2, 4, 3, 5)
        print("PASS dtype", dtype)
    print("local_semantic_check: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
