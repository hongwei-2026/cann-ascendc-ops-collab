# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P025 样例 **PASS**；已合入 PR#1 座位代码） |
| Owner | hongwei-2026 |
| 更新日期 | 2026-11-03 |
| 队友 | `people/yuhongming-2026/` 已合入（PR#1 / P022；八参 Select 仍未云测，勿当默认试交包） |
| 基线 | EXP_P025：Build OK；`dx`/`update` mismatched=**0/192** |
| 代码 | Compare(+zero)+count Select；默认仅 `DT_FLOAT` |
| 试交包 | `deliverables/PReluGradUpdate_2329_code_best.zip`（P025） |
| 下一闸 | 开 fp16/bf16（无 Cast 路径）或平台试交本包 |
| 依据 | `explorations/logs/EXP_P025_20261103_125446/` · EXP012 |
