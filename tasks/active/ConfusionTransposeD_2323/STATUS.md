# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`性能中 / 平台RE待诊断`**（P004本机22/22通过；平台点1/2约2.4×改善、3～5未改善；整次Runtime Error，未过验收） |
| Owner | yuhongming-2026（原功能实现）；**hongwei-2026**（P002 云测 + 性能座位） |
| 更新日期 | 2026-11-10 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | [EXP_C004](explorations/EXP_C004-contiguous-row-dma.md)：P004 样例 PASS + 原13/新9共22/22 bitwise PASS（含 bf16/int64）；此前 EXP_C001/C002 保留 |
| 性能基线 | [EXP_C003](explorations/EXP_C003-perf-baseline-platform.md)：平台 1～5 Pass，用时 ~1.7ms / 最优 ~3.5μs（~500×） |
| 下一闸 | **P005仅诊断**：原版/P004边界探测+主机阶段计时；不改Host/Kernel，见 [EXP_C005](explorations/EXP_C005-platform-re-and-boundary-diagnostic.md)、[DEC002](decisions/DEC002-platform-runtime-error-diagnostic-first.md) |
| 止损 | `docs/collaboration/02-perf-gate.md`；触线写 `decisions/DEC*` |

## 一句话

P004有有限收益，但平台整次Runtime Error；失败点和根因未知，先诊断再决定下一条性能路线。
