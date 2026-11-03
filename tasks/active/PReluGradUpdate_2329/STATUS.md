# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P025 样例 **PASS**；低精度待开） |
| Owner | hongwei-2026 |
| 更新日期 | 2026-11-03 |
| 基线 | EXP_P025：Build OK；`dx`/`update` mismatched=**0/192** |
| 代码 | Compare(+zero)+count Select；默认仅 `DT_FLOAT` |
| 试交包 | `deliverables/PReluGradUpdate_2329_code_best.zip`（P025） |
| 下一闸 | 开 fp16/bf16（无 Cast 路径）或平台试交本包 |
| 依据 | `explorations/logs/EXP_P025_20261103_125446/` · EXP012 |
