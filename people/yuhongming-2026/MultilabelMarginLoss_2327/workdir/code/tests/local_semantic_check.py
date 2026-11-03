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


if __name__ == "__main__":
    check_case_seed42_mean()
    check_none_sum_shapes()
    check_count_truncation()
    print("MultilabelMarginLoss local semantic checks: PASS")
