#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""ConfusionTransposeD golden：permute + reshape 融合（与 torch.permute/view 对齐）。"""
import numpy as np

try:
    from ml_dtypes import bfloat16
except ImportError:  # pragma: no cover
    bfloat16 = None


def impl(x, perm, shape, transpose_first):
    if isinstance(perm, np.ndarray):
        perm = perm.tolist()
    if isinstance(shape, np.ndarray):
        shape = shape.tolist()
    if isinstance(transpose_first, np.ndarray):
        transpose_first = transpose_first.item()

    perm = list(perm)
    shape = list(shape)
    transpose_first = bool(transpose_first)

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if transpose_first:
        # permute → contiguous → reshape(=view)
        y = np.ascontiguousarray(np.transpose(x, axes=perm)).reshape(shape)
    else:
        y = np.ascontiguousarray(np.reshape(x, shape))
        y = np.ascontiguousarray(np.transpose(y, axes=perm))
    return y


if __name__ == "__main__":
    x1 = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(np.float32)
    y1 = impl(x1, [0, 2, 1, 3], [2, 4, 3, 5], True)
    assert y1.shape == (2, 4, 3, 5)
    assert np.allclose(y1, np.transpose(x1, (0, 2, 1, 3)))
    y2 = impl(x1, [0, 2, 1, 3], [2, 4, 3, 5], False)
    assert y2.shape == (2, 3, 4, 5)
    print("reference self-check OK", y1.shape, y2.shape)
