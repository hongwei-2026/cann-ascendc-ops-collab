# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | **性能优化中：P105 云测 PASS；平台计时待测** |
| Owner | hongwei-2026 |
| 队友 | yuhongming-2026 · JadeYara（见 TEAM_NOTICE.md） |
| 更新日期 | 2026-11-04 |
| 当前源码 | P105 tip `2e378eb`：权重准备前移 DeQue 前；Kernel LF `4f100a73…3520c` |
| 本地检查 | weight/compute/semantic PASS；6 ops / 4 PIPE_V 同 P103 |
| 云测 | **EXP_P105_204700**：verify PASS；三 key Done；dx/update **0/192 PASS** |
| 平台 P104 | 19:59:09 Pass；40.42 分；不采纳 |
| 对照 P103 | 历史 7.36/7.30/7.24/5.83/6.07 μs，42.89 |
| 本轮试交 | **P105** → deliverables/SUBMIT_LINKS_P105.md |
| 下一闸 | 平台 vs P103；无 ≥2% → 回 P103 + DEC |
| 依据 | EXP025 · path-ledger P105 |

## 给队友一句话

P105 云测功能已过（0/192）。现交平台计时；无收益回 P103。入口：SUBMIT_LINKS_P105.md。
