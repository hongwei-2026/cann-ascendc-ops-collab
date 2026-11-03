# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P030：修 case4 507035） |
| Owner | hongwei-2026 |
| 更新日期 | 2026-11-03 |
| 基线 | P029 云测 PASS；平台 1–3 Pass，4 仍 507035 |
| 代码 | half Cast↔fp32；bf16 位转换+同步；三 dtype |
| 下一闸 | CodeArts `-e` → 平台重交 |
| 依据 | EXP017 |
