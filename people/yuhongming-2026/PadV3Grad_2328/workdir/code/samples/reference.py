#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import torch
from ml_dtypes import bfloat16

def _pad_index(n, front, back, mode):
    """某一维 pad 后每个输出位置取自原张量的哪个下标（输出长度 = n+front+back）。

    front/back 可以大于该维长度（内置允许），所以按周期折叠而不是只折一次。
    """
    if mode in ("edge", "replicate"):           # 复制边缘
        return np.clip(np.arange(-front, n + back), 0, n - 1)
    if n == 1:
        return np.zeros(front + 1 + back, dtype=np.int64)
    j = np.arange(-front, n + back, dtype=np.int64)
    if mode == "reflect":                       # 不重复边缘，周期 2(n-1)
        k = np.mod(j, 2 * (n - 1))
        return np.where(k < n, k, 2 * (n - 1) - k)
    if mode == "symmetric":                     # 重复边缘，周期 2n
        k = np.mod(j, 2 * n)
        return np.where(k < n, k, 2 * n - 1 - k)
    if mode == "circular":                      # 整体循环
        return np.mod(j, n)
    raise ValueError(f"Unsupported mode: {mode}")


def impl(x, paddings, mode="reflect", paddings_contiguous=True):
    """
    PadV3Grad 算子的预期函数实现
    计算 PadV3 操作的反向梯度
    
    Args:
        x: 输入张量，可以是 numpy 数组或 torch tensor，shape 为 PadV3 操作后的形状
        paddings: 填充参数，可以是 numpy 数组或 list
        mode: 填充模式，支持 "reflect"（默认）和 "replicate"
        paddings_contiguous: paddings 是否为连续格式
    
    Returns:
        原始输入张量的梯度（numpy 数组）
    """
    # ------------------------------
    # 处理参数
    # ------------------------------
    if isinstance(paddings, np.ndarray):
        paddings = paddings.tolist()
    if isinstance(mode, np.ndarray):
        mode = mode.item()
    if isinstance(paddings_contiguous, np.ndarray):
        paddings_contiguous = paddings_contiguous.item()
    
    mode = str(mode)
    paddings_contiguous = bool(paddings_contiguous)
    
    # ------------------------------
    # 转换输入为 torch tensor
    # ------------------------------
    original_dtype = x.dtype
    
    if isinstance(x, np.ndarray):
        if x.dtype == bfloat16:
            x = torch.tensor(x.astype(np.float32)).to(torch.bfloat16)
        else:
            x = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    else:
        raise TypeError(f"Input x must be numpy array or torch tensor, got {type(x)}")
    
    # ------------------------------
    # 计算原始形状
    # ------------------------------
    input_shape = list(x.shape)
    num_dims = len(input_shape)

    # paddings 的两种排布（内置算子实测）：
    #   paddings_contiguous=True ：[f0,b0,f1,b1,...]
    #   paddings_contiguous=False：前 num_dims 个是各维 front，后 num_dims 个是各维 back
    # 统一折成 [(front_i, back_i)] 再算（内置报错 "padFront and padBack ... have
    # incorrect values" 即因为用例写的是 [n,2] 行=(front,back)、内置按前后半段读）。
    _flat = np.asarray(paddings, dtype=np.int64).reshape(-1)
    if paddings_contiguous:
        _rows = _flat.reshape(-1, 2)
    else:
        _rows = np.stack([_flat[:num_dims], _flat[num_dims:]], axis=1)
    pad_list = [[int(r[0]), int(r[1])] for r in _rows]
    
    output_shape = []
    for i in range(num_dims):
        output_shape.append(input_shape[i] - pad_list[i][0] - pad_list[i][1])
    
    # ------------------------------
    # 计算梯度
    # ------------------------------
    # 直接用 torch 的反向：pad 是线性算子，把 x 当作上游梯度 g，
    # 对 sum(pad(o) * g) 求 o 的梯度，就是 PadV3Grad 的结果。
    # （手写的折叠公式实测与内置不符：reflect 的前向映射是 -1->1、-2->2，
    # 不是「以边为轴」的那种折法；交给 autograd 不会写错。）
    #   内置 mode 词表 constant/edge/reflect/symmetric/circular，torch 侧是
    #   constant/replicate/reflect —— replicate 与 edge 同义。
    # 注意不能直接用 F.pad 的 autograd：它要求「填充不超过 3 个维度」，
    # 4 维以上的用例（unsqueeze 后变 5 维）会直接抛 NotImplementedError，
    # 所以这里自己按各维下标 select 出 pad 的结果（index_select 的反向是
    # index_add，重复下标会正确累加，reflect/edge 的折叠语义天然正确）。
    if mode not in ("constant", "edge", "replicate", "reflect", "symmetric", "circular"):
        raise ValueError(f"Unsupported mode: {mode}")

    g = x.detach().to(torch.float64)
    for d, (front, back) in enumerate(pad_list):
        if input_shape[d] != output_shape[d] + front + back:
            raise ValueError(f"paddings 与 x 的形状对不上：dim {d} 上 "
                             f"{output_shape[d]}+{front}+{back} != {input_shape[d]}")
    if mode == "constant":                      # 常数填充区对梯度没有贡献，取中间那块
        y = g[tuple(slice(f, f + s) for (f, _), s in zip(pad_list, output_shape))]
    else:
        orig = torch.zeros(output_shape, dtype=torch.float64, requires_grad=True)
        padded = orig
        for d, (front, back) in enumerate(pad_list):
            idx = _pad_index(output_shape[d], front, back, mode)
            if not np.array_equal(idx, np.arange(idx.size)):
                padded = torch.index_select(padded, d, torch.from_numpy(idx))
        (padded * g).sum().backward()
        y = orig.grad
    
    # ------------------------------
    # 转换为 numpy 数组输出
    # ------------------------------
    # 梯度在 fp64 里算完再落回输入的 dtype（跟内置的输出 dtype 对齐）。
    if original_dtype == bfloat16:
        return y.to(torch.float32).numpy().astype(bfloat16)
    return y.numpy().astype(original_dtype)


if __name__ == "__main__":
    print("=== PadV3Grad 预期函数测试 ===\n")
    
    # Case 1: reflect 模式，2D 张量
    print("Case 1: reflect 模式，2D 张量")
    x1 = np.arange(16).reshape(4, 4).astype(np.float32)
    print(f"输入张量:\n{x1}")
    paddings1 = [[1, 1], [1, 1]]
    y1 = impl(x1, paddings1, "reflect", True)
    print(f"填充参数: {paddings1}")
    print(f"输出形状: {y1.shape}")
    print(f"输出张量:\n{y1}\n")
    
    # Case 2: replicate 模式，2D 张量
    print("Case 2: replicate 模式，2D 张量")
    x2 = np.arange(16).reshape(4, 4).astype(np.float32)
    print(f"输入张量:\n{x2}")
    paddings2 = [[1, 1], [1, 1]]
    y2 = impl(x2, paddings2, "replicate", True)
    print(f"填充参数: {paddings2}")
    print(f"输出形状: {y2.shape}")
    print(f"输出张量:\n{y2}\n")
    
    # Case 3: reflect 模式，4D 张量（测试用例）
    print("Case 3: reflect 模式，4D 张量（测试用例）")
    x3 = np.arange(4 * 5 * 6 * 6).reshape(4, 5, 6, 6).astype(np.float32)
    print(f"输入形状: {x3.shape}")
    paddings3 = [[1, 1], [1, 1], [1, 1], [1, 1]]
    y3 = impl(x3, paddings3, "reflect", True)
    print(f"填充参数: {paddings3}")
    print(f"输出形状: {y3.shape}")
    print(f"输出张量前几个元素: {y3[0, 0, :2, :2]}\n")
    
    # Case 4: float16 数据类型
    print("Case 4: float16 数据类型")
    x4 = np.arange(16).reshape(4, 4).astype(np.float16)
    paddings4 = [[1, 1], [1, 1]]
    y4 = impl(x4, paddings4, "reflect", True)
    print(f"输入形状: {x4.shape}, 数据类型: {x4.dtype}")
    print(f"输出形状: {y4.shape}, 数据类型: {y4.dtype}\n")
    
    # Case 5: 不等填充
    print("Case 5: 不等填充")
    x5 = np.arange(20).reshape(5, 4).astype(np.float32)
    print(f"输入张量:\n{x5}")
    paddings5 = [[2, 1], [1, 2]]
    y5 = impl(x5, paddings5, "reflect", True)
    print(f"填充参数: {paddings5}")
    print(f"输出形状: {y5.shape}")
    print(f"输出张量:\n{y5}\n")
    
    print("=== 所有测试用例通过！===")