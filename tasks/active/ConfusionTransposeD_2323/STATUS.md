# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`性能中`**（P004 云端样例 + 22/22 bitwise PASS；同机连续行加速 2.09～6.07×，平台待重提） |
| Owner | yuhongming-2026（原功能实现）；**hongwei-2026**（P002 云测 + 性能座位） |
| 更新日期 | 2026-11-10 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | [EXP_C004](explorations/EXP_C004-contiguous-row-dma.md)：P004 样例 PASS + 原13/新9共22/22 bitwise PASS（含 bf16/int64）；此前 EXP_C001/C002 保留 |
| 性能基线 | [EXP_C003](explorations/EXP_C003-perf-baseline-platform.md)：平台 1～5 Pass，用时 ~1.7ms / 最优 ~3.5μs（~500×） |
| 下一闸 | **重提平台验证 4761239 代码**；P004 已采纳有限收益，≥10×假设未达；冻结本轮 Kernel，见 [DEC001](decisions/DEC001-adopt-p004-and-verify-platform.md) |
| 止损 | `docs/collaboration/02-perf-gate.md`；触线写 `decisions/DEC*` |

## 一句话

P004 保精度并取得连续行 2.09～6.07× 同机收益；平台约 500× 差距的归因仍待核定，下一闸平台重提。
