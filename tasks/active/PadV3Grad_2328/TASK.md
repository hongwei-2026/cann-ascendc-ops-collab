# 题目：PadV3Grad

## 元信息

| 项 | 内容 |
|----|------|
| 题目 ID | 2328 |
| 类型 | A 算子（开源仓算子工程 / ict） |
| 硬件 | 昇腾 NPU（模板注册 ascend910b） |
| 对标 | 昇腾内置 PadV3Grad 的 TBE 实现 |
| 模板 | `PadV3Grad_problem_2328_template.zip`，原文在 `materials/_unzipped/code/` |
| 实现座位 | `people/yuhongming-2026/PadV3Grad_2328/workdir/code/` |

## 验收摘要

- 用 Ascend C 实现与模板 `reference.py` 一致的 Pad 反向。
- 数据类型：float16、float32、bfloat16，ND。paddings 为 int32 或 int64。
- mode：constant、edge、replicate、reflect、symmetric、circular。
- `paddings_contiguous` 为真时按 `(front, back)` 成对存放；为假时前半是各维 front，后半是各维 back。
- 精度：float `1e-4`，half `1e-3`。bf16 走 float32 累加后再转回。

## 公式

`y` 的每一维长度是 `x` 的该维减去 front 和 back。

- constant：丢掉填充区，只保留中间。
- 其它模式：填充后的下标 `j = coord - front`，再按 reference 的周期规则映回 `[0, n)`，把 `x` 上落到同一位置的值相加。

## 非目标

- 本机没有 NPU，不把 `build.sh -e` 写成通过。
- 不改其他题目，不改他人 `people/` 目录。
