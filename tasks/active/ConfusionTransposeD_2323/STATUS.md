# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`实现中 / P010`**（针对平台 RE + 连续行 UB 饿死；无 Debug 探针） |
| Owner | yuhongming-2026（原功能）；**hongwei-2026**（性能座位） |
| 更新日期 | 2026-11-15 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | EXP_C001/C002/C004；P010 待云测/平台 |
| 性能基线 | 平台点1～5 ~1.3–2.0ms vs 最优 ~3.5μs；同机 TBE ≈4.804μs；P004 曾点1/2≈717μs |
| 下一闸 | CodeArts：样例 + 39 bitwise + 故意超 UB case 应 Pass；再提平台看点6 RE 是否消失。**禁** CTD_PROBE |
| 止损 | `docs/collaboration/02-perf-gate.md`；触线写 DEC |

## 一句话

P010：超 UB 改为 GM gather（治 RE），连续行 DMA 吃满 UB（治饿死）。不宣称追平 TBE；见 [EXP_C010](explorations/EXP_C010-spill-and-row-ub.md)。
