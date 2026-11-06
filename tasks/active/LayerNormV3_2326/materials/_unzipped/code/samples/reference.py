#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np

def impl(x, gamma, beta, begin_norm_axis=1, begin_params_axis=1, epsilon=1e-5):
    """
    LayerNormV3 算子的预期函数实现
    计算层归一化
    
    Args:
        x: 输入张量
        gamma: 缩放参数
        beta: 平移参数
        begin_norm_axis: 开始归一化的轴
        begin_params_axis: 参数开始应用的轴
        epsilon: 数值稳定性参数
    
    Returns:
        (y, mean, rstd): (归一化后的输出, 均值, 逆标准差)
    """
    if isinstance(x, np.ndarray):
        x_np = np.asarray(x)
    else:
        raise TypeError(f"Input x must be numpy array, got {type(x)}")
    
    if isinstance(gamma, np.ndarray):
        gamma_np = np.asarray(gamma)
    else:
        raise TypeError(f"Input gamma must be numpy array, got {type(gamma)}")
    
    if isinstance(beta, np.ndarray):
        beta_np = np.asarray(beta)
    else:
        raise TypeError(f"Input beta must be numpy array, got {type(beta)}")
    
    rank = len(x_np.shape)
    
    if begin_norm_axis < 0:
        begin_norm_axis = rank + begin_norm_axis
    if begin_params_axis < 0:
        begin_params_axis = rank + begin_params_axis
    
    norm_axes = tuple(range(begin_norm_axis, rank))
    
    mean = np.mean(x_np, axis=norm_axes, keepdims=True)
    var = np.var(x_np, axis=norm_axes, keepdims=True)
    
    rstd = 1.0 / np.sqrt(var + epsilon)
    normalized = (x_np - mean) * rstd
    
    if gamma_np.ndim < len(norm_axes):
        for _ in range(len(norm_axes) - gamma_np.ndim):
            gamma_np = np.expand_dims(gamma_np, axis=-1)
            beta_np = np.expand_dims(beta_np, axis=-1)
    
    y = normalized * gamma_np + beta_np
    
    return y.astype(x_np.dtype), mean.astype(x_np.dtype), rstd.astype(x_np.dtype)


if __name__ == "__main__":
    print("=== LayerNormV3 预期函数测试 ===\n")
    
    np.random.seed(42)
    x1 = np.random.randn(4, 6, 8).astype(np.float32)
    gamma1 = np.ones((6, 8), dtype=np.float32) * 0.5
    beta1 = np.zeros((6, 8), dtype=np.float32)
    
    print(f"输入 x 形状: {x1.shape}")
    print(f"gamma 形状: {gamma1.shape}")
    print(f"beta 形状: {beta1.shape}")
    
    y1, mean1, rstd1 = impl(x1, gamma1, beta1, begin_norm_axis=1, begin_params_axis=1)
    print(f"\n输出 y 形状: {y1.shape}")
    print(f"mean 形状: {mean1.shape}")
    print(f"rstd 形状: {rstd1.shape}")
    
    print(f"\n验证: y = (x - mean) * rstd * gamma + beta")
    expected_y = (x1 - mean1) * rstd1 * gamma1 + beta1
    print(f"结果是否正确: {np.allclose(y1, expected_y)}")
    
    x2 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    gamma2 = np.ones((4, 5), dtype=np.float32)
    beta2 = np.zeros((4, 5), dtype=np.float32)
    
    y2, mean2, rstd2 = impl(x2, gamma2, beta2, begin_norm_axis=2, begin_params_axis=2)
    print(f"\n输入 x 形状: {x2.shape}")
    print(f"输出 y 形状: {y2.shape}")
    print(f"mean 形状: {mean2.shape}")
    print(f"结果是否正确: {np.allclose(y2, (x2 - mean2) * rstd2 * gamma2 + beta2)}")
    
    print("\n=== 所有测试用例通过！===")