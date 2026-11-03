# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P026：开三 dtype、禁 Cast，待云测） |
| Owner | hongwei-2026 |
| 更新日期 | 2026-11-03 |
| 基线 | P025 样例 PASS；平台 1–3 Pass，**4 点 ACL 507035** |
| 代码 | Compare(+zero)+count Select；fp16/bf16 无 Cast；workspace 16MB |
| 下一闸 | CodeArts verify → `-e`（须三 key 编过）→ 平台重交 |
| 依据 | EXP012 · EXP013 |
