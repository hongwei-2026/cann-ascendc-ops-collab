# 状态 — QuantUpdateScatter_2330

| 项 | 内容 |
|----|------|
| 状态 | **`性能中` / 精度修复中** |
| Owner | yuhongming-2026 |
| 更新日期 | 2026-11-15 |
| 代码座位 | `people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/` |
| Q001 云冒烟 | PASS `@6a7b849` samples `-e`（小 shape） |
| 平台评测 | 测点1 Pass（5.89us vs 最优 3.68us）；**测点2 Runtime Error / ACL 161002**；3–5 Skipped |
| R001 | OpDef 补齐 4 签名 + kernel 按 `indicesIsInt64` 分发（待平台重测） |
| 下一闸 | 云端 `-e` 再冒烟 → **重新上传平台** 看测点2 |

## 一句话

Q001 性能路径保留；平台 RE 优先按 **缺 dtype/indices 组合注册** 修（R001），根因待重测确认。
