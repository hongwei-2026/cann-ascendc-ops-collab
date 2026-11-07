#!/usr/bin/env python3
"""CPU regression for MultilabelMarginLoss reference semantics."""

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
    # 首个 <0 前缀计数：与手工扫描一致
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


def check_out_of_range_does_not_truncate():
    # 9 不在 [0, C)，跳过但不截断；-1 才截断。
    x = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    target = np.array([[3, 9, 1, -1, 0]], dtype=np.int32)
    y, is_target = reference.impl(x, target, reduction="none")
    np.testing.assert_array_equal(is_target[0], [0, 1, 0, 1, 0, 0])
    # 手算：l in {1,3}，j in {0,2,4,5}
    # loss = 1/6 * sum_l sum_j max(0, 1+x[j]-x[l])
    manual = 0.0
    for ell in (1, 3):
        for j in (0, 2, 4, 5):
            manual += max(0.0, 1.0 + float(x[0, j]) - float(x[0, ell]))
    manual /= 6.0
    np.testing.assert_allclose(y, [manual], rtol=1e-6, atol=1e-6)


def check_empty_prefix_is_zero():
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    target = np.array([[-1, 0, 1, 2], [-5, 3, 3, 1]], dtype=np.int32)
    y, is_target = reference.impl(x, target, reduction="none")
    np.testing.assert_array_equal(is_target, np.zeros((2, 4), dtype=np.int32))
    np.testing.assert_array_equal(y, np.zeros(2, dtype=np.float32))


def check_duplicate_weight_and_half():
    x = np.array([[0.5, -1.0, 2.0, 0.25]], dtype=np.float32)
    target = np.array([[1, 1, 0, -1]], dtype=np.int32)
    y, is_target = reference.impl(x, target, reduction="sum")
    np.testing.assert_array_equal(is_target[0], [1, 2, 0, 0])
    # cnt[1]=2, cnt[0]=1; 非目标 j={2,3}
    manual = 0.0
    for ell, cnt in ((0, 1), (1, 2)):
        for j in (2, 3):
            manual += cnt * max(0.0, 1.0 + float(x[0, j]) - float(x[0, ell]))
    manual /= 4.0
    np.testing.assert_allclose(y, [manual], rtol=1e-6, atol=1e-6)

    xh = x.astype(np.float16)
    yh, _ = reference.impl(xh, target, reduction="none")
    assert yh.dtype == np.float16
    np.testing.assert_allclose(yh.astype(np.float32), np.float32(manual), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    check_case_seed42_mean()
    check_none_sum_shapes()
    check_count_truncation()
    check_out_of_range_does_not_truncate()
    check_empty_prefix_is_zero()
    check_duplicate_weight_and_half()
    print("MultilabelMarginLoss local semantic checks: PASS")
