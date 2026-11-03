# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | **P103 本地候选已实现；CANN 构建与平台计时待测**（P100 为当前实测参考） |
| Owner | hongwei-2026 |
| 队友 | yuhongming-2026 · JadeYara（见 [TEAM_NOTICE.md](TEAM_NOTICE.md)） |
| 更新日期 | 2026-11-03 |
| tip | 当前 Kernel 是基于 `3bd65fb` P100 的 P103 本地候选；P102 历史云测 tip `2e3cbc6` |
| 最近云测 | 历史 EXP_P102_154833：三 key Done + 样例 **0/192 PASS**（kernel LF `0e58c2b2…`）；P103 待云测 |
| 最近平台数据 | 2026-11-03 15:48:12 历史截图：**1–5 Pass**，42.06 分，8.89/8.75/8.99/6.11/6.01 μs；对应源码未由截图 hash 证明，不作为 P103 数据 |
| 默认代码 | `people/hongwei-2026/.../workdir/code/`（**P103 本地候选**；P100 是已测性能参考） |
| 队友近况 | yuhongming **P032**：座位吸对齐 Duplicate + `workspace=0`；**尚未**吸收 P100 全核/`Muls`/progress 补丁；仍含八参 Select（勿交平台） |
| 下一闸 | 先对 P103 做 CANN 构建和样例检查，再绑定源码/产物指纹做同环境平台计时，与 P100 比较 |
| 依据 | [TEAM_NOTICE](TEAM_NOTICE.md) · EXP019 · EXP022 · [DEC001](decisions/DEC001-p102-rollback.md) · EXP_P102 |

## 给队友一句话

P102 已 DEC 不采纳。席位当前是 **P103** 本地候选（零缓冲复用/缩短屏障）；**P100 仍是已测性能参考**。先云测再平台 A/B，勿叠新微优化。
