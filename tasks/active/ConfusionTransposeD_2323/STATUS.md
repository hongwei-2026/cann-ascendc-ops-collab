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
| 下一闸 | **P007已止损，当前回到P004 `4761239`**：C007在910B3/CANN9.1.0 exit=0，P004/P007均39/39 bitwise PASS，4项profile全有效；但128行只有1.024×（未达1.3×），小样慢6.86%（超过2%回退上限），因此DEC006关闭P007。当前无后续性能路线授权；平台Runtime Error仍未知 |
| 止损 | `docs/collaboration/02-perf-gate.md`；触线写 `decisions/DEC*` |

## 一句话

P006与P007均已按门槛止损，活动Kernel恢复到P004 `4761239`。C007虽39/39 bitwise通过，128行目标仅1.024×且小样回退6.86%，不采纳P007；当前没有下一条优化路线授权。平台Runtime Error仍未定位。
