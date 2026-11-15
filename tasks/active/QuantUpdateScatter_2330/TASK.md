# 题目：QuantUpdateScatter

## 元信息

| 项 | 内容 |
|----|------|
| 题目 ID | 2330 |
| 类型 | A 算子（AscendC / ops-nn 方向） |
| 硬件 | ascend910b（模板已注册） |
| 对标 | 内置 AscendC `quant_update_scatter`（CAST_RINT 路径） |
| 来源 ZIP | `D:\华为算子比赛\QuantUpdateScatter_2330.zip`（不入库） |
| 实现座位 | `people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/` |

## 语义（摘要）

```text
y = var.copy()   # inplace 同址可跳过拷贝
for p, idx in enumerate(indices):          # indices 长度对齐 updates.dim0
    for k in range(updates.shape[axis]):   # axis 仅 -2
        y[p, ..., idx+k, :] = clip_int8_rint(
            fp32(updates[p,...,k,:]) / fp32(scales) + fp32(zp))
```

- `quant_axis` 仅 `-1`；`reciprocal_scale=true` 时改为乘。
- 量化**不读** var 原值，只覆盖 indices 指定行；其余保持 var。
- dtype 组：MODE_0 = int8 / int32 / fp16 / fp32 / int32；MODE_1 = int8 / int64 / bf16 / bf16 / bf16。

## 验收注意

- samples 首测点极小 `(1,2,32)`，只作功能冒烟，**不是**性能门禁 case。
- 精度：与 reference / 内置一致（RINT + clip）；勿改成 fp32→half→int8 二次舍入。
- 性能门禁 shape / TBE 对照：**尚未入库**，云测后补 `explorations/`。
