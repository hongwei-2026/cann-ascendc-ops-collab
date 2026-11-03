# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P031：修平台 case4 `ACL 507035`） |
| Owner | hongwei-2026 |
| 队友 | yuhongming-2026 · JadeYara（见 [TEAM_NOTICE.md](TEAM_NOTICE.md)） |
| 更新日期 | 2026-11-03 |
| tip | 工作树 P031（待提交、待 CodeArts） |
| 云测 | P029 三 key + 样例 **0/192 PASS** |
| 平台 | 1–3 **Pass**；**4 RE 507035**；5 Skipped |
| 代码座位 | `people/hongwei-2026/.../workdir/code/`（修正 per-channel weight UB 起点对齐） |
| 下一闸 | CodeArts 验证 P031 → 平台重交 |
| 依据 | [TEAM_NOTICE](TEAM_NOTICE.md) · EXP018 · EXP_P029 |

## 给队友一句话

平台第 4 点向量核异常定位到更直接的地址约束：`innerSize` 非 8-float 倍数时，旧代码从非 32B 对齐的 `weightLocal` 偏移启动 `Duplicate`。P031 已改为标量补齐未对齐前缀、向量只从对齐地址启动；待 CodeArts 与平台确认。
