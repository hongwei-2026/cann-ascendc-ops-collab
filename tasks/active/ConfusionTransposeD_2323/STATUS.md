# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`诊断中 / 探针探形`**（P004 再提仍 RE；P008 TBE≈4.804μs；下一闸见 [HANDOFF-PROBE-SHAPES.md](HANDOFF-PROBE-SHAPES.md)） |
| Owner | yuhongming-2026（原功能）；**hongwei-2026**（性能座位） |
| 更新日期 | 2026-11-12 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | EXP_C001/C002/C004：样例+泛化 bitwise PASS |
| 性能基线 | 平台 ~1.7ms vs 最优 ~3.5μs；**同机系统 TBE last10 mean=4.804μs** @ `(2,1024,128)`（`ConfusionTransposeD` / `te_confusiontransposed_*`）；P004 平台点1/2 ~717μs |
| 下一闸 | **P009 Host诊断探针待CodeArts验证**：infer/tiling记录输入、属性、输出及拒绝阶段；本机语义与模拟序列化已通过。先执行[C009云测](C009-CLOUD-RUN.md)，再判断平台是否可回收日志；平台点1～6的shape目前均未知。微优化仍按 DEC007 冻结 |
| 止损 | `docs/collaboration/02-perf-gate.md` |

## 一句话

P004 再提：1/4/5≈720μs、2/3≈2.0ms、点6隐藏、整次 RE。系统 TBE≈4.8μs。P009 Host探针待云端功能及日志验证，平台真实输入与RE原因仍未知。
