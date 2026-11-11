# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`性能中 / 平台RE待诊断`**（P004本机22/22通过；平台点1/2约2.4×改善、3～5未改善；整次Runtime Error，未过验收） |
| Owner | yuhongming-2026（原功能实现）；**hongwei-2026**（P002 云测 + 性能座位） |
| 更新日期 | 2026-11-11 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | [EXP_C004](explorations/EXP_C004-contiguous-row-dma.md)：P004 样例 PASS + 原13/新9共22/22 bitwise PASS（含 bf16/int64）；此前 EXP_C001/C002 保留 |
| 性能基线 | [EXP_C003](explorations/EXP_C003-perf-baseline-platform.md)：平台 1～5 Pass，用时 ~1.7ms / 最优 ~3.5μs（~500×） |
| 下一闸 | **P007候选待CodeArts验证**：C006确认P006步长DMA对5个定向布局均变慢，rank2和小样持平；已结案并回到P004 `4761239`。P007仅尝试把P004连续行中源地址等差的多行合为一次MTE2步长DMA；`local_semantic_check`、P004计划1658项和P007计划266项均通过，目标案例模型搬运调用128→32。实测门槛仍是`dma_rows128_f32`≥1.3×、rank2/小样回退≤2%；Owner需运行云端build、39项bitwise及profile。门槛未达即DEC止损，不再开启下一路线。平台RE仍未知 |
| 止损 | `docs/collaboration/02-perf-gate.md`；触线写 `decisions/DEC*` |

## 一句话

P006已在910B3/CANN9.1.0上39/39 bitwise通过，但目标span=1路径普遍变慢，按DEC005关闭并以P004为生产起点。当前只推进一次有时限的P007连续行批量DMA假设；CodeArts和平台Runtime Error仍待后续证据，不把本地profile等同平台表现。
