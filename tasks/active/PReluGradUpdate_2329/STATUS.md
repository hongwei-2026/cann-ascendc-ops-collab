# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P025：仅 fp32 待云测） |
| Owner | hongwei-2026 |
| 更新日期 | 2026-11-03 |
| 基线 | EXP_P024：verify PASS；仍 `not support bf16 type cast` |
| 代码 | Compare(+zero)+count Select；默认只注册/编 `DT_FLOAT` |
| 下一闸 | pull → verify（须见 DataType FLOAT-only）→ `rm -rf build` → `-e` |
| 依据 | `explorations/logs/EXP_P024_20261103_124542/` · EXP011 |
