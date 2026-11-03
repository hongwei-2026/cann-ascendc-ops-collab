# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `性能优化中`（P100：去 16MB 空 workspace + 全核 channel tile） |
| Owner | hongwei-2026 |
| 队友 | yuhongming-2026 · JadeYara（见 [TEAM_NOTICE.md](TEAM_NOTICE.md)） |
| 更新日期 | 2026-11-03 |
| tip | 工作树 P100（待 CodeArts） |
| 云测 | P029/P030 三 key + 样例 **0/192 PASS** |
| 平台 | **1–5 Pass**；用时 1.30/1.32/2.00/1.79/1.79ms，参考 13.76/13.73/13.72/9.82/10.21μs |
| 代码座位 | `people/hongwei-2026/.../workdir/code/`（P100 保留 P031 正确性） |
| 下一闸 | CodeArts 三 key + `-e` → 平台采时；未测前不开双缓冲 |
| 依据 | 平台截图 2026-11-03 · EXP018 · EXP019 |

## 给队友一句话

P100 方向对（去 16MB workspace + 全核 + channel 内 `Muls`），但原稿 `progress += tileLen_` 在 channel 截断时会跳元素；已改 `progress += currentNum`，Host `tileLen <= innerSize`。先云测再交平台采时。
