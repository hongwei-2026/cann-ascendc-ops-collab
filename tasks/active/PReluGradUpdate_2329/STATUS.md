# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P031 平台 1–5 **Pass**；性能未达标） |
| Owner | hongwei-2026 |
| 队友 | yuhongming-2026 · JadeYara（见 [TEAM_NOTICE.md](TEAM_NOTICE.md)） |
| 更新日期 | 2026-11-03 |
| tip | `5aad4c0`（P031） |
| 云测 | P029/P030 三 key + 样例 **0/192 PASS** |
| 平台 | **1–5 Pass**（含原 case4）；6 隐藏；用时 ~1.3–2.0ms vs 最优 ~4μs |
| 代码座位 | `people/hongwei-2026/.../workdir/code/`（weight Duplicate 32B 对齐） |
| 下一闸 | 性能有界优化（带宽题；勿小 shape 死磕） |
| 依据 | 平台截图 2026-11-03 · EXP018 |

## 给队友一句话

P031 已验证：case4 `ACL 507035` 消失，可见点 1–5 全过。当前瓶颈是性能（约慢 300×+）；功能勿再乱改对齐逻辑。
