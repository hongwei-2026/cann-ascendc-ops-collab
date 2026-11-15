# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`待平台验证 / P010`**（云测大 tensor bitwise PASS；无 Debug） |
| Owner | yuhongming-2026（原功能）；**hongwei-2026**（性能座位） |
| 更新日期 | 2026-11-15 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/`（`bc14e41`） |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | 样例 PASS；npu_regression 13/13；8MiB 0/2097152 |
| 性能基线 | 平台曾 ~1.3–2.0ms vs 最优 ~3.5μs；TBE≈4.804μs；P010 平台成绩待测 |
| 下一闸 | **提平台 P010**（无 CTD_PROBE）：看点6 RE 是否消失、点1～5用时。微优化/TBE 镜像仍另案 |
| 止损 | `docs/collaboration/02-perf-gate.md` |

## 一句话

P010 云测通过（含 8MiB 超 UB 正确）。下一闸平台验证隐藏点 RE 与性能。
