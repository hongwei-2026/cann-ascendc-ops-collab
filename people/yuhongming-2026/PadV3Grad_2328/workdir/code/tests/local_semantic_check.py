#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""对照 samples/reference.py 检查 PadV3Grad 的折回下标。"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "samples"))
import reference


def py_mod(a, m):
    return np.mod(a, m)


def map_coord(coord, front, n, mode):
    j = coord - front
    if mode in ("edge", "replicate"):
        if n <= 1:
            return 0
        return int(np.clip(j, 0, n - 1))
    if n <= 1:
        return 0
    if mode == "reflect":
        period = 2 * (n - 1)
        k = int(py_mod(j, period))
        return k if k < n else period - k
    if mode == "symmetric":
        period = 2 * n
        k = int(py_mod(j, period))
        return k if k < n else period - 1 - k
    return int(py_mod(j, n))


def fold_grad(x, front, back, mode):
    """按 kernel 的逐维累加，在 float64 里求和。constant 直接切中间。"""
    x = np.asarray(x, dtype=np.float64)
    rank = x.ndim
    if mode == "constant":
        slices = tuple(slice(front[d], front[d] + (x.shape[d] - front[d] - back[d])) for d in range(rank))
        return x[slices].copy()
    cur = x.copy()
    cur_shape = list(cur.shape)
    for axis in range(rank):
        if front[axis] == 0 and back[axis] == 0:
            continue
        extent = cur_shape[axis]
        n = extent - front[axis] - back[axis]
        moved = np.moveaxis(cur, axis, 0)
        acc = np.zeros((n,) + moved.shape[1:], dtype=np.float64)
        for s in range(extent):
            acc[map_coord(s, front[axis], n, mode)] += moved[s]
        cur = np.moveaxis(acc, 0, axis)
        cur_shape[axis] = n
    return cur


def parse_pads(x, paddings, contiguous):
    flat = np.asarray(paddings, dtype=np.int64).reshape(-1)
    rank = x.ndim
    if contiguous:
        rows = flat.reshape(-1, 2)[:rank]
    else:
        rows = np.stack([flat[:rank], flat[rank: rank * 2]], axis=1)
    return rows[:, 0].tolist(), rows[:, 1].tolist()


def check_against_reference(x, paddings, mode, contiguous=True, atol=1e-6, rtol=1e-6):
    front, back = parse_pads(x, paddings, contiguous)
    got = fold_grad(np.asarray(x, dtype=np.float64), front, back, mode)
    exp = reference.impl(np.asarray(x), paddings, mode, contiguous)
    np.testing.assert_allclose(got, np.asarray(exp, dtype=np.float64), atol=atol, rtol=rtol)


def test_modes_and_layouts():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1, 1, 8, 8)).astype(np.float32)
    pads = [[0, 0], [0, 0], [2, 2], [2, 2]]
    for mode in ("reflect", "constant", "edge", "replicate", "symmetric", "circular"):
        check_against_reference(x, pads, mode)

    x2 = rng.normal(size=(1, 1, 7, 10)).astype(np.float32)
    check_against_reference(x2, [[0, 0], [0, 0], [1, 2], [3, 0]], "reflect", True)
    check_against_reference(x2, [0, 0, 1, 3, 0, 0, 2, 0], "reflect", False)

    # pad 大于该维长度，按周期折。
    wide = rng.normal(size=(13,)).astype(np.float32)
    check_against_reference(wide, [[5, 5]], "reflect")
    check_against_reference(wide, [[5, 5]], "symmetric")
    check_against_reference(wide, [[5, 5]], "circular")
    check_against_reference(wide, [[5, 5]], "edge")

    # 输出长度为 1：reflect 的周期不能为 0，所有位置都回到这一个元素。
    one = rng.normal(size=(9,)).astype(np.float32)
    check_against_reference(one, [[4, 4]], "reflect")

    rank1 = rng.normal(size=(6,)).astype(np.float32)
    check_against_reference(rank1, [[1, 1]], "replicate")
    check_against_reference(rank1, [[0, 0]], "circular")


def test_half_and_bf16():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(2, 3, 6, 5)).astype(np.float32)
    pads = [[0, 0], [1, 0], [1, 2], [2, 1]]
    folded = fold_grad(x, [0, 1, 1, 2], [0, 0, 2, 1], "reflect")
    half = reference.impl(x.astype(np.float16), pads, "reflect", True)
    np.testing.assert_allclose(half.astype(np.float32), folded.astype(np.float16).astype(np.float32), atol=1e-3, rtol=1e-3)
    bf = reference.impl(x.astype(bfloat16), pads, "symmetric", True)
    expect = fold_grad(x, [0, 1, 1, 2], [0, 0, 2, 1], "symmetric").astype(bfloat16)
    np.testing.assert_allclose(bf.astype(np.float32), expect.astype(np.float32), atol=1e-2, rtol=1e-2)


def test_sample_manual():
    x = np.arange(64, dtype=np.float32).reshape(1, 1, 8, 8)
    y = reference.impl(x, [[0, 0], [0, 0], [2, 2], [2, 2]], "reflect", True)
    assert y.shape == (1, 1, 4, 4)
    # 两维都是 front=2、n=4 的 reflect，下标 0 只由填充后下标 2 折回。
    np.testing.assert_allclose(y[0, 0, 0, 0], x[0, 0, 2, 2])
    folded = fold_grad(x, [0, 0, 2, 2], [0, 0, 2, 2], "reflect")
    np.testing.assert_allclose(folded, y.astype(np.float64))


if __name__ == "__main__":
    test_modes_and_layouts()
    test_half_and_bf16()
    test_sample_manual()
    print("PadV3Grad local semantic checks: PASS")
