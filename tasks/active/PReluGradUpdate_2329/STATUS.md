# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | **性能优化中**（P104 平台变慢 → **不采纳**；默认回 P103） |
| Owner | hongwei-2026 |
| 队友 | yuhongming-2026 · JadeYara（见 [TEAM_NOTICE.md](TEAM_NOTICE.md)） |
| 更新日期 | 2026-11-04 |
| 默认 tip | **P103 `47db93e`**（平台 **42.89**） |
| 云测 | P103 / P104 均为 0/192 PASS |
| 平台 P104 | 19:59 提交 1132450 Pass；**8.15/8.22/8.25/5.79/5.99 μs**（前三点劣于 P103） |
| 平台 P103 | 16:46：**7.36/7.30/7.24/5.83/6.07 μs**，42.89 |
| 默认试交 | 按 **P103** 源码交；席位 workdir 或仍为 P104 候选供 Codex 续改 |
| 下一闸 | Codex 新道路；先 ledger；P104 见 DEC002 |
| 依据 | [DEC002](decisions/DEC002-p104-platform-regression.md) · EXP024 · 截图 EXP024-platform-195909 |

## 给队友一句话

**P104 平台变差，不采纳。** 默认仍用 **P103 / 42.89**。性能相对 TBest 仍不够。
