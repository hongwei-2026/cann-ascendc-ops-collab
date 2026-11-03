# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P027：fp32+fp16 无 bf16，待云测） |
| Owner | hongwei-2026 |
| 更新日期 | 2026-11-03 |
| 基线 | EXP_P026：开三 dtype 仍 `not support bf16 type cast` |
| 代码 | Compare(+zero)；fp16 标量转换；**bf16 关闭** |
| 下一闸 | verify → `-e`（应只编 2 key）→ 平台重交 |
| 依据 | EXP_P026 · EXP014 |
