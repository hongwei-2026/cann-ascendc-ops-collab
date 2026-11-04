# 状态 — MultilabelMarginLoss_2327

| 项 | 内容 |
|----|------|
| 状态 | **`性能中`（入口）**：平台用例 1–5 功能 Pass；耗时远慢于最优 |
| Owner | 共享台账；实现座位见下 |
| 更新日期 | 2026-11-04 |
| 代码座位 | **hongwei M009 候选（未云测）**：`people/hongwei-2026/MultilabelMarginLoss_2327/workdir/code/`；已通过基线仍为 M006 |
| 队友验证座 | yuhongming 同源 tip 本地 `-e` PASS |
| 设计 | [SUMMARY](analysis/SUMMARY.md) · [design_final](analysis/AI-codex-claude_design_final.md) · [design.md](design.md) |
| 本地闸 | EXP_M003 / EXP_M004：**PASSED** |
| 平台闸 | [用例 1–5 Pass](explorations/PLATFORM_CASES_1-5_20261104.md)（错误 0%；用时 ~713µs vs 最优 ~5µs） |
| 下一闸 | 按 `02-perf-gate.md` 开有界性能优化；先同机最优/TBE 对照，禁止小 shape 无限抠 |
| 本轮候选 | [M009 按需工作区](explorations/EXP_M009-workspace.md)：CPU 检查通过；CANN 编译/NPU/性能均未测，原 M006 ZIP 保留 |
| 禁止串题 | 勿与 `PReluGradUpdate_2329` 共用 build/EXP 根目录 |

## 一句话

功能精度已过平台可见点；下一阶段性能（约 150× 差距），先记账再优化。
