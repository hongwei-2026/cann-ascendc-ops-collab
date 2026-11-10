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
| 下一闸 | **P005结案，停止追加同批日志诊断**：candidate六项已确认为Host拒绝（UB限制3、跨rank1、-1属性2），均ACL561002；两版同结果，未观察到新增失败。平台RE根因仍未知，性能未达标。后续功能边界修复和性能路线分别新立P行，按 [DEC004](decisions/DEC004-close-c005-host-rejections.md) 分流，不混入本次归档 |
| 止损 | `docs/collaboration/02-perf-gate.md`；触线写 `decisions/DEC*` |

## 一句话

P004有有限收益；C005六项本地失败已由Host明确拒绝解释，未启动Kernel。平台Runtime Error仍未定位，性能目标仍未完成；停止反复取证，后续修复须另立有界路线。
