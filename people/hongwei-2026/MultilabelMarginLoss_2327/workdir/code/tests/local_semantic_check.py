#!/usr/bin/env python3
"""CPU regression for MultilabelMarginLoss reference semantics (+ classTile mirror)."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "samples"))
import reference  # noqa: E402


def check_case_seed42_mean():
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, [4, 6]).astype(np.float32)
    target = rng.integers(-1, 5, size=[4, 6], dtype=np.int32)
    y, is_target = reference.impl(x, target, reduction="mean")
    assert y.shape == (1,)
    assert is_target.shape == (4, 6)
    assert is_target.dtype == np.int32
    for i in range(4):
        cnt = np.zeros(6, dtype=np.int32)
        for t in target[i]:
            if t < 0:
                break
            if 0 <= t < 6:
                cnt[t] += 1
        np.testing.assert_array_equal(is_target[i], cnt)


def check_none_sum_shapes():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    target = np.array([[1, 2, -1, 0], [0, 0, 3, -1], [2, -1, 1, 1]], dtype=np.int32)
    y_none, _ = reference.impl(x, target, reduction="none")
    y_sum, _ = reference.impl(x, target, reduction="sum")
    y_mean, _ = reference.impl(x, target, reduction="mean")
    assert y_none.shape == (3,)
    assert y_sum.shape == (1,)
    assert y_mean.shape == (1,)
    np.testing.assert_allclose(y_mean, y_sum / 3.0, rtol=1e-5, atol=1e-5)


def check_count_truncation():
    x = np.zeros((1, 6), dtype=np.float32)
    target = np.array([[3, 2, 0, 3, 0, 0]], dtype=np.int32)
    _, is_target = reference.impl(x, target, reduction="none")
    np.testing.assert_array_equal(is_target[0], [3, 0, 1, 2, 0, 0])

    target2 = np.array([[3, 3, 3, 3, -1, 1]], dtype=np.int32)
    _, is_target2 = reference.impl(x, target2, reduction="none")
    np.testing.assert_array_equal(is_target2[0], [0, 0, 0, 4, 0, 0])


def check_k_ne_c_and_any_neg_truncate():
    """K≠C；截断条件是首个 <0（不限 -1）。"""
    x = np.arange(10, dtype=np.float32).reshape(2, 5)
    target = np.array([[1, 4, -2, 0, 9, 3], [0, -1, 2, 2, 2, 2]], dtype=np.int32)
    y, is_target = reference.impl(x, target, reduction="none")
    assert is_target.shape == (2, 5)
    np.testing.assert_array_equal(is_target[0], [0, 1, 0, 0, 1])
    np.testing.assert_array_equal(is_target[1], [1, 0, 0, 0, 0])
    assert y.shape == (2,)


def check_empty_positive_mean():
    """无正类：loss=0，仍计入 mean 的 N。"""
    x = np.ones((3, 4), dtype=np.float32)
    target = np.array([[-1, 0, 1, 2], [-5, 1, 1, 1], [9, 9, -1, 0]], dtype=np.int32)
    y_mean, is_target = reference.impl(x, target, reduction="mean")
    y_none, _ = reference.impl(x, target, reduction="none")
    assert np.all(is_target == 0)
    np.testing.assert_allclose(y_none, 0.0)
    np.testing.assert_allclose(y_mean, [0.0])


def class_tile_mirror(x, target, class_tile, reduction="mean"):
    """Mirror kernel sliding classTile hinge (fp32), for equivalence vs reference."""
    xf = np.asarray(x, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.int32)
    n, c = xf.shape
    k = tgt.shape[1]
    losses = np.zeros(n, dtype=np.float32)
    is_target = np.zeros((n, c), dtype=np.int32)

    for i in range(n):
        prefix = k
        for idx, t in enumerate(tgt[i]):
            if t < 0:
                prefix = idx
                break
        cnt = np.zeros(c, dtype=np.int32)
        for t in tgt[i, :prefix]:
            if 0 <= t < c:
                cnt[t] += 1
        is_target[i] = cnt
        if not np.any(cnt > 0):
            continue
        loss = 0.0
        for j0 in range(0, c, class_tile):
            j_len = min(class_tile, c - j0)
            xjf = xf[i, j0 : j0 + j_len] + 1.0
            mask = (cnt[j0 : j0 + j_len] == 0).astype(np.float32)
            if not np.any(mask):
                continue
            acc = np.zeros(j_len, dtype=np.float32)
            for l0 in range(0, c, class_tile):
                l_len = min(class_tile, c - l0)
                xlf = xf[i, l0 : l0 + l_len]
                cl = cnt[l0 : l0 + l_len]
                for ll in range(l_len):
                    cc = int(cl[ll])
                    if cc == 0:
                        continue
                    tmp = np.maximum(0.0, xjf - xlf[ll])
                    if cc != 1:
                        tmp = tmp * float(cc)
                    acc = acc + tmp
            loss += float((acc * mask).sum())
        losses[i] = loss / float(c)

    if reduction == "mean":
        y = np.atleast_1d(np.mean(losses)).astype(xf.dtype)
    elif reduction == "sum":
        y = np.atleast_1d(np.sum(losses)).astype(xf.dtype)
    else:
        y = losses.astype(xf.dtype)
    return y, is_target


def check_class_tile_mirror_matches_reference():
    rng = np.random.default_rng(7)
    for c, tile, k in [(6, 8, 6), (17, 8, 11), (33, 16, 20)]:
        n = 5
        x = rng.uniform(-3, 3, [n, c]).astype(np.float32)
        target = rng.integers(-2, c + 2, size=[n, k], dtype=np.int32)
        y_ref, it_ref = reference.impl(x, target, reduction="mean")
        y_m, it_m = class_tile_mirror(x, target, class_tile=tile, reduction="mean")
        np.testing.assert_array_equal(it_m, it_ref)
        np.testing.assert_allclose(y_m, y_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    check_case_seed42_mean()
    check_none_sum_shapes()
    check_count_truncation()
    check_k_ne_c_and_any_neg_truncate()
    check_empty_positive_mean()
    check_class_tile_mirror_matches_reference()
    print("MultilabelMarginLoss local semantic checks: PASS")
