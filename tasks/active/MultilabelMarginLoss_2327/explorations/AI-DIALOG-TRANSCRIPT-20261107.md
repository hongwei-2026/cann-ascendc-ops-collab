# 对话记录 — MultilabelMarginLoss_2327 — 2026-11-07

用户：按 LayerNormV3 同样的座位、台账、对话、提交和链接流程，参考 `MultilabelMarginLoss_problem_2327_template.zip`，把 MultilabelMarginLoss 做完整。本机能测的测，没有 NPU 就不测设备。

## 题目

与昇腾内置 TBE MultilabelMarginLoss 对齐：`x` 为 float / fp16 / bf16 的 `[N, C]`，`target` 为 int32 `[N, K]`。从 `target` 头部扫描，遇到第一个 `<0` 截断；只统计 `0 <= t < C` 的出现次数。`is_target` 是这个计数。行损失为

`1/C * sum_{j: cnt[j]==0} sum_{l: cnt[l]>0} cnt[l] * max(0, 1 + x[j] - x[l])`

`reduction`：`none` 输出 `[N]`，`mean`（默认）和 `sum` 输出 `[1]`。`y` 的 dtype 跟 `x`。比较是逻辑值，不是二进制。精度门槛沿用平台默认：float `1e-4`，half `1e-3`，整数 `0`。

模板 zip 里的 kernel、tiling、InferShape 仍是 `TODO`。`samples/reference.py` 已经是上面的计数语义，没有改。

## 这次改了什么

`people/yuhongming-2026` 座位里原先已有功能实现，本地 910B4 样例曾通过。它仍有两处和当前结构对不上：

1. 小形状的 mean/sum 也会按多核切，并始终申请 16MB workspace。多核 `SyncAll` 如果没有框架系统 GM 前缀，云上会报 ACL 507035。
2. kernel 单测还在写已删除的 `totalNum` / `blockFactor` / `ubFactor`，对当前 `MultilabelMarginLossTilingData` 编不过。

这次把已核对过的修法放进本座位，没有改 hongwei 目录：

- `none`，或实际只用 1 个核时，workspace 为 0，行损失在核内按 float32 累加后写 `y`，不走 `SyncAll`。
- 多核 mean/sum 才申请 16MB 系统区，再加按 8 对齐的用户行损失。`N <= 8`、`C <= 32`、`K <= 32` 且 `N*C*K <= 4096` 的 mean/sum 强制 1 核。
- Host 单测改成带 `reduction` 的样例、none、bf16 sum、`N=100` 多核、单行 fp16。样例期望是 `4 6 6 1 1 4 4 6 6 `，workspace `{0}`。多核期望 workspace `16777632`。
- kernel 单测改成真实 tiling 字段，1 个 block、`usedCores=1`。

## 本机测试

跑了 `tests/local_semantic_check.py`，对照 `reference.py`：

- 样例 mean
- none / sum / mean 的形状
- 正类重复计数，以及 `<0` 截断
- 越界正标签只跳过、不截断，并与手算 hinge 对上
- 前缀一开始就是负数时，`is_target` 和 loss 都是 0
- 重复标签按次数加权
- fp16 结果落在 `1e-3` 内

输出：`MultilabelMarginLoss local semantic checks: PASS`。

本机没有 CANN / NPU，没有跑 `build.sh -e`。不把这次写成设备 PASS。平台性能差距（约 150 倍）这次没有改算法，仍记在原台账。
