# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`性能中`**（功能/泛化已 PASS；平台用时 ~1.7ms vs 最优 ~3.5μs） |
| Owner | yuhongming-2026（原功能实现）；**hongwei-2026**（P002 云测 + 性能座位） |
| 更新日期 | 2026-11-10 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | [EXP_C001](explorations/EXP_C001-local-910b4-pass.md) 样例 PASS；[EXP_C002](explorations/EXP_C002-generalize-cloud.md) 13/13 PASS |
| 性能基线 | [EXP_C003](explorations/EXP_C003-perf-baseline-platform.md)：平台 1～5 Pass，用时 ~1.7ms / 最优 ~3.5μs（~500×） |
| 下一闸 | **P004 连续行 DMA 候选云测**：原 13 + 新 9 项 bitwise 回归、原版/候选同机 msprof；当前只有 CPU 语义验证，未采纳，见 [EXP_C004](explorations/EXP_C004-contiguous-row-dma.md) |
| 止损 | `docs/collaboration/02-perf-gate.md`；触线写 `decisions/DEC*` |

## 一句话

功能已绿；标量 gather 导致平台性能约慢 500×，进入性能专题（hongwei 座位）。
