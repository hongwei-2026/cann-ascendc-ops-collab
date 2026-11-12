# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`性能中 / P008对照未完成`**（活动 Kernel=P004 `4761239`；P006/P007 止损；平台 RE 仍未知） |
| Owner | yuhongming-2026（原功能实现）；**hongwei-2026**（P002 云测 + 性能座位） |
| 更新日期 | 2026-11-12 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | [EXP_C004](explorations/EXP_C004-contiguous-row-dma.md)：P004 样例 PASS + 原13/新9共22/22 bitwise PASS（含 bf16/int64）；此前 EXP_C001/C002 保留 |
| 性能基线 | [EXP_C003](explorations/EXP_C003-perf-baseline-platform.md)：平台 1～5 Pass，用时 ~1.7ms / 最优 ~3.5μs（~500×）；P004 同机连续行 2～6×，平台点1/2约2.4×、3～5无改善 |
| 下一闸 | **系统可调用已证实**（Host 0.0953ms @ sample）。下一步：同 shape 再测一档更大输入的系统 Host 时间，并（若 vendor 包已在 build）对比自定义 P004；要 device 时间再用 msprof。禁止新 Kernel / 平台粘贴 |
| 止损 | `docs/collaboration/02-perf-gate.md`；触线写 `decisions/DEC*` |

## 一句话

P006/P007 已止损，活动 Kernel 仍是 P004。C008 已证实公开 API 属性对齐且 TBE 走 AscendC 六路分派，但**没有**同机系统 TBE profile；当前不能当「新优化版」粘贴平台，也未授权新 Kernel 路线。
