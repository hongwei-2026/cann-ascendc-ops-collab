# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P030：修平台 case4 `ACL 507035`） |
| Owner | hongwei-2026 |
| 队友 | yuhongming-2026 · JadeYara（见 [TEAM_NOTICE.md](TEAM_NOTICE.md)） |
| 更新日期 | 2026-11-03 |
| tip | `4d0fdb5`（P030） |
| 云测 | P029 三 key + 样例 **0/192 PASS** |
| 平台 | 1–3 **Pass**；**4 RE 507035**；5 Skipped |
| 代码座位 | `people/hongwei-2026/.../workdir/code/`（half Cast↔fp32；bf16 位转换+同步） |
| 下一闸 | CodeArts 验证 P030 → 平台重交 |
| 依据 | [TEAM_NOTICE](TEAM_NOTICE.md) · EXP017 · EXP_P029 |

## 给队友一句话

平台第 4 点向量核异常还在；P029 开齐 dtype 不够。正在试 **half 走 Cast、bf16 补同步**（P030）。详情看 TEAM_NOTICE，勿抢改 hongwei 目录。
