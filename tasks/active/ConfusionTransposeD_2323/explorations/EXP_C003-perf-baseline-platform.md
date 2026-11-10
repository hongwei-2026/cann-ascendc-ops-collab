# EXP_C003 — 平台性能基线（功能 PASS / 性能 ~500× 慢）

| 字段 | 内容 |
|------|------|
| 题目 | ConfusionTransposeD_2323 |
| 日期 | 2026-11-10 |
| 作者 | hongwei-2026 |
| 阶段 | 性能基线 |
| 结论 | **采纳为性能基线**（功能绿；性能不可接受） |
| 座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| path-ledger | P003 起点 |

## 平台结果（截图归档意图）

| 测试点 | 结果 | 错误占比 | 用时 | 最优用时 |
|--------|------|----------|------|----------|
| 1 | Pass | 0.00% | 1.76ms | 3.66μs |
| 2 | Pass | 0.00% | 1.75ms | 3.45μs |
| 3 | Pass | 0.00% | 1.73ms | 3.45μs |
| 4 | Pass | 0.00% | 1.70ms | 3.50μs |
| 5 | Pass | 0.00% | 1.82ms | 3.43μs |
| 6 | 隐藏 | — | — | — |

相对最优约 **480～530×**。

## 根因假设（待优化验证）

Kernel `ComputeTile` 使用 `MapSrcIndex` + `GetValue`/`SetValue` 标量循环；无连续块拷 / 无向量 gather / 无 Transpose 原语。

## 下一闸

按 [HANDOFF-PERF-PROMPT.md](../HANDOFF-PERF-PROMPT.md) 开性能道路；遵守 `docs/collaboration/02-perf-gate.md`。
