#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""LayerNormV3 参考实现，对齐 TBE：总体方差，gamma/beta 按尾维广播。"""
import numpy as np


def impl(x, gamma, beta, begin_norm_axis=1, begin_params_axis=1, epsilon=1e-5):
    """
    y = (x - mean) * rstd * gamma + beta
    mean、var 在 [begin_norm_axis, rank) 上计算，var 除以 N（ddof=0）。
    rstd = 1 / sqrt(var + epsilon)
    gamma/beta 与 x 的尾维 numpy 广播；begin_params_axis 用来声明 gamma 覆盖的起始轴。
    计算用 float32。y 按 x 的 dtype 回写，mean/rstd 按 gamma 的 dtype 回写。
    """
    x_np = np.asarray(x)
    gamma_np = np.asarray(gamma)
    beta_np = np.asarray(beta)
    rank = x_np.ndim
    if begin_norm_axis < 0:
        begin_norm_axis += rank
    if begin_params_axis < 0:
        begin_params_axis += rank
    if begin_norm_axis < 0 or begin_norm_axis > rank:
        raise ValueError("begin_norm_axis out of range")
    if begin_params_axis < 0 or begin_params_axis > rank:
        raise ValueError("begin_params_axis out of range")

    xf = x_np.astype(np.float32, copy=False)
    gf = gamma_np.astype(np.float32, copy=False)
    bf = beta_np.astype(np.float32, copy=False)
    axes = tuple(range(begin_norm_axis, rank))
    mean = np.mean(xf, axis=axes, keepdims=True) if axes else xf
    if not axes:
        mean = np.array(xf, dtype=np.float32, copy=True)
        var = np.zeros_like(mean)
    else:
        var = np.mean(np.square(xf - mean), axis=axes, keepdims=True)
    rstd = 1.0 / np.sqrt(var + np.float32(epsilon))
    y = (xf - mean) * rstd * gf + bf
    return y.astype(x_np.dtype), mean.astype(gamma_np.dtype), rstd.astype(gamma_np.dtype)


if __name__ == "__main__":
    np.random.seed(42)
    x1 = np.random.randn(4, 6, 8).astype(np.float32)
    gamma1 = np.ones((6, 8), dtype=np.float32) * 0.5
    beta1 = np.zeros((6, 8), dtype=np.float32)
    y1, mean1, rstd1 = impl(x1, gamma1, beta1, begin_norm_axis=1, begin_params_axis=1)
    expected = (x1 - mean1) * rstd1 * gamma1 + beta1
    assert np.allclose(y1, expected)
    assert mean1.shape == (4, 1, 1)

    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    gamma2 = np.random.randn(4).astype(np.float32)
    beta2 = np.random.randn(4).astype(np.float32)
    y2, mean2, rstd2 = impl(x2, gamma2, beta2, begin_norm_axis=2, begin_params_axis=2)
    assert y2.shape == x2.shape and mean2.shape == (2, 3, 1)
    assert np.allclose(y2, (x2 - mean2) * rstd2 * gamma2 + beta2)

    y3, mean3, _ = impl(x2, gamma2, beta2, begin_norm_axis=-1, begin_params_axis=-1)
    assert np.allclose(y3, y2) and np.allclose(mean3, mean2)
    print("reference self-check passed")
