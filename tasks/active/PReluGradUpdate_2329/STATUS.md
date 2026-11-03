# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | **性能优化中**（P103 已作默认；公开点仍慢于 TBest） |
| Owner | hongwei-2026 |
| 队友 | yuhongming-2026 · JadeYara（见 [TEAM_NOTICE.md](TEAM_NOTICE.md)） |
| 更新日期 | 2026-11-03 |
| tip | P103 `47db93e` / 台账含平台图 |
| 云测 | EXP_P103_163813：三 key Done + **0/192 PASS** |
| 平台 | 16:46:42：**1–5 Pass**；**42.89** 分；7.36/7.30/7.24/5.83/6.07 μs（TBest ~4.0–4.2μs） |
| 默认代码 | `people/hongwei-2026/.../workdir/code/`（**P103**） |
| 对比 | 相对 P102 图（42.06 / ~8.xμs）明显回升；相对 P100 图（42.41 / ~7.xμs）分数更高、耗时近似 |
| 下一闸 | 有界下一条或 B 类仿真/分析；勿无限抠公开小 shape |
| 依据 | EXP023 · EXP_P103 · 截图 `explorations/screenshots/EXP023-platform-20261103-164642.png` |

## 给队友一句话

默认用 hongwei **P103**（42.89）。P102 已废弃。公开点仍约 1.5–1.8× TBest，新道路先写 path-ledger。
