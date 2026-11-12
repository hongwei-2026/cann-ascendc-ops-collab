# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`性能暂停 / 等 Owner 决策`**（活动 Kernel=P004 `4761239`；P006/P007/Host 基线止损；**P008 device TBE 基线已立**） |
| Owner | yuhongming-2026（原功能）；**hongwei-2026**（性能座位） |
| 更新日期 | 2026-11-12 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | EXP_C001/C002/C004：样例+泛化 bitwise PASS |
| 性能基线 | 平台 ~1.7ms vs 最优 ~3.5μs；**同机系统 TBE device ≈4.7–4.9μs** @ `(2,1024,128)`（`te_confusiontransposed_*`）；P004 平台点1/2 ~717μs |
| 下一闸 | [DEC007](decisions/DEC007-p008-tbe-device-baseline.md)：停微优化；要么 Owner 批准镜像 TBE 模板结构大改，要么收口转平台 RE。**禁**平台粘未验证新包 |
| 止损 | `docs/collaboration/02-perf-gate.md` |

## 一句话

系统 TBE 在同机已到 **~5μs** 量级（与榜首同阶）；P004 微优化追不上。Kernel 冻结，等结构大改立项或性能收口。
