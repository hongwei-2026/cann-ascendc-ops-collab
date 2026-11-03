# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P024：关 bf16 key + DtypeBridge，待云测） |
| Owner | hongwei-2026 |
| 更新日期 | 2026-11-03 |
| 基线 | EXP_P023：`not support bf16 type cast`（同 EXP002） |
| 代码 | Compare(+zero)+count Select；默认不注册/不编 bf16 |
| 试交包 | 待 P024 云编通过后再打 |
| 下一闸 | pull → verify PASS → `rm -rf build` → `-e` |
| 依据 | `explorations/logs/EXP_P023_20261103_123326/` · EXP010 |
