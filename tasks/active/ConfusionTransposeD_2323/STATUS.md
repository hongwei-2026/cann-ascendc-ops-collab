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
| 下一闸 | **P009平台提交（禁新增文件）**：CTD_PROBE 已内联进既有 Host `tiling`/`infershape`，勿再上传 `*_probe.h`。验证平台日志有无 `CTD_PROBE`；耗时不作成绩；微优化仍冻 |
| 止损 | `docs/collaboration/02-perf-gate.md` |

## 一句话

P004 再提：1/4/5≈720μs、2/3≈2.0ms、点6隐藏、整次 RE。系统 TBE≈4.8μs。P009修订版通过云端tiling探针和功能检查，平台真实输入与RE原因仍未知；见 [EXP_C009](explorations/EXP_C009-platform-shape-probe.md)。
