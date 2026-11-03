# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | **`性能优化中`**（功能 1–5 已 Pass） |
| Owner | hongwei-2026 |
| 队友 | yuhongming-2026 · JadeYara（见 [TEAM_NOTICE.md](TEAM_NOTICE.md)） |
| 更新日期 | 2026-11-03 |
| tip | P100 已云测基线 `3bd65fb` / 台账 `8762e74`；当前 P102 候选未云测 |
| 云测 | EXP_P100：三 key Done + 样例 **0/192 PASS** |
| 平台 | 截图提交 13557230920：**1–5 Pass**；42.41 分；7.26/7.41/7.33/5.72/5.82 μs（源码 hash 未显示） |
| 默认代码 | `people/hongwei-2026/.../workdir/code/`（P100 基线；P102 对齐搬运候选待云测） |
| 队友近况 | yuhongming **P032**：座位吸对齐 Duplicate + `workspace=0`；**尚未**吸收 P100 全核/`Muls`/progress 补丁；仍含八参 Select（勿交平台） |
| 下一闸 | P102 先做 CodeArts 三 key 编译和样例，再同机平台复测；无增益按性能门禁止损 |
| 依据 | [TEAM_NOTICE](TEAM_NOTICE.md) · EXP018 · EXP019 · EXP022 · EXP_P100 |

## 给队友一句话

现在不是纠 case4 了，是**性能**。默认用 hongwei P100；P032 对齐请留在自己座位，不要带回八参 Select，也不要抢改 hongwei / `code_best`。
