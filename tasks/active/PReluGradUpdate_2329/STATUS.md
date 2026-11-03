# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | **`性能复测中`**（P102 未见收益，已恢复 P100） |
| Owner | hongwei-2026 |
| 队友 | yuhongming-2026 · JadeYara（见 [TEAM_NOTICE.md](TEAM_NOTICE.md)） |
| 更新日期 | 2026-11-03 |
| tip | 当前 Kernel 与语义检查已恢复 `3bd65fb` 的 P100；P102 历史云测 tip `2e3cbc6` |
| 云测 | EXP_P102_154833：三 key Done + 样例 **0/192 PASS**（kernel LF `0e58c2b2…`） |
| 平台 | 2026-11-03 15:48:12 截图：**1–5 Pass**，42.06 分，8.89/8.75/8.99/6.11/6.01 μs；比上一图五点均慢；截图不含源码 hash |
| 默认代码 | `people/hongwei-2026/.../workdir/code/`（**P100**；已撤回 P102 对齐分支，恢复后的平台计时待复测） |
| 队友近况 | yuhongming **P032**：座位吸对齐 Duplicate + `workspace=0`；**尚未**吸收 P100 全核/`Muls`/progress 补丁；仍含八参 Select（勿交平台） |
| 下一闸 | 锁定源码与构建产物，同环境对 P100/历史 P102 重复 A/B；获得 profile 前不叠加新微优化 |
| 依据 | [TEAM_NOTICE](TEAM_NOTICE.md) · EXP019 · EXP022 · [DEC001](decisions/DEC001-p102-rollback.md) · EXP_P102 |

## 给队友一句话

席位默认已恢复 hongwei **P100**；P102 本轮不采纳。历史云测日志保留；恢复 P100 不等于平台耗时已复现。当前只在本地修改，不推送。
