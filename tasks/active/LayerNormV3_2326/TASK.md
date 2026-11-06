# 题目：LayerNormV3

## 元信息

| 项 | 内容 |
|----|------|
| 题目 ID | 2326 |
| 类型 | A 算子（开源仓算子工程 / ict） |
| 硬件 | 昇腾 NPU（模板注册 ascend910b） |
| CANN | 9.0.0 或 9.1.0（模板 README） |
| 对标 | 昇腾内置 LayerNormV3 的 TBE 实现 |
| 模板 | `LayerNormV3_problem_2326_template.zip`，原文在 `materials/_unzipped/code/` |
| 实现座位 | `people/yuhongming-2026/LayerNormV3_2326/workdir/code/` |

## 验收摘要（抄自题面）

### 功能

- 用 Ascend C 实现与内置 LayerNormV3 TBE 核心功能一致的算子，完成设计、开发、测试。
- 支持模板给出的数据类型与格式：float32 / float16 / bfloat16，ND。
- 比较从二进制比较改为逻辑值比较。
- 必须泛化：合法的轴、shape、gamma/beta 尾维广播都要能算。

### 精度

- float：1e-4
- half：1e-3
- 题面平台默认句含 int 阈值 0；本算子 def 没有整数输出。

### 性能

- 题面要求功能、精度、性能一起验。本会话先做正确的按行实现，不把未跑的性能写成通过。

### 测试

- 参考内置 TBE 自建全场景用例。样例点是模板 `samples/`（x `[4,6,8]`，gamma/beta `[6,8]`，`begin_norm_axis=1`）。
- 无 NPU 的泛化自测：`tests/local_semantic_check.py`。
- 设备上：`bash build.sh -e`。

## 计算公式

`mean`、`var` 沿 `[begin_norm_axis, rank)`，`var` 除以 N。

`rstd = 1 / sqrt(var + epsilon)`（默认 `1e-5`）

`y = (x - mean) * rstd * gamma + beta`

`mean`/`rstd` 形状与 x 相同，但从 `begin_norm_axis` 起的维度为 1。gamma/beta 的元素个数等于 `x[begin_params_axis:]`。

## 非目标

- 本会话不改其他题目，不改他人 `people/` 目录。
- 不在没有 910B 日志的情况下宣称 `build.sh -e` 已通过。
- 不照搬 ops-nn 里那套 Welford 大 kernel。

## 参考

- 题面答疑：https://gitcode.com/cann/ops-nn/issues
- 本仓同类落地方式：`tasks/active/CtcLossV2_2325/`、`people/yuhongming-2026/CtcLossV2_2325/`
