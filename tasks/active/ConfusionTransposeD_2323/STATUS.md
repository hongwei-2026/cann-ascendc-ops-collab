# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`诊断中 / P010 平台仍 RE`**（云测大 tensor PASS；平台点1～5 Pass，点6/7/8 隐藏） |
| Owner | yuhongming-2026（原功能）；**hongwei-2026**（性能座位） |
| 更新日期 | 2026-11-15 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/`（P010 / `bc14e41`） |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | 云测样例+13/13+8MiB bitwise PASS；平台点1～5 错误 0% |
| 性能基线 | 平台 P010：1.26/1.24/1.93/1.77/1.73ms vs 最优 ~3.4–3.6μs；同机 TBE≈4.804μs |
| 下一闸 | RE 未因去超 UB 拒绝而消失。优先：下载工程日志；CodeArts 扩 rank/-1/numel 边界。禁 Debug 上平台；微优化/TBE 镜像另批 |
| 止损 | `docs/collaboration/02-perf-gate.md` |

## 一句话

P010 平台仍 Runtime Error；公开点正确但 ~1.2–1.9ms；隐藏点现为 **6/7/8**。超 UB 假说不足以单独解释 RE。
