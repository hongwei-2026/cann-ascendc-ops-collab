# 状态 — ConfusionTransposeD_2323

| 项 | 内容 |
|----|------|
| 状态 | **`诊断中 / 平台 RE 未解`**（干净 P004 再提仍 RE；平台禁 Debug 探针；P008 TBE≈4.804μs） |
| Owner | yuhongming-2026（原功能）；**hongwei-2026**（性能座位） |
| 更新日期 | 2026-11-12 |
| 代码座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| 模板 | `materials/ConfusionTransposeD_problem_2323_template.zip` |
| 功能闸 | EXP_C001/C002/C004：样例+泛化 bitwise PASS |
| 性能基线 | 平台 ~1.7ms vs 最优 ~3.5μs；**同机系统 TBE last10 mean=4.804μs** @ `(2,1024,128)`（`ConfusionTransposeD` / `te_confusiontransposed_*`）；P004 平台点1/2 ~717μs |
| 下一闸 | 干净 P004 @19:32 仍 RE：点1～5 Pass（1.26～2.02ms），点6隐藏。平台禁探针→无法从平台日志取 shape。可选：下载工程日志 / CodeArts 扩边界猜点6 / Owner 批 TBE 模板大改。微优化仍冻 |
| 止损 | `docs/collaboration/02-perf-gate.md` |

## 一句话

干净 P004 再提仍 Runtime Error；点1～5 正确但 ~1.3–2.0ms（最优 ~3.5μs）。平台拒 Debug，点6 shape 仍未知。见 [C009-platform-clean-p004](explorations/logs/EXP_C009/C009-platform-clean-p004-20261112.md)。
