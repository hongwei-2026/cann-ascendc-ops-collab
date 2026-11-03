# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P028：回到仅 fp32；低精度另开） |
| Owner | hongwei-2026 |
| 更新日期 | 2026-11-03 |
| 基线 | P025 样例 PASS；平台 1–3 Pass；P027 开 fp16 仍 Cast 炸 |
| 代码 | Compare(+zero)；**仅 DT_FLOAT** |
| 试交 | 桌面/deliverables `code_best.zip`（fp32-only） |
| 下一闸 | 平台继续交 fp32 包；或专开 half 独立实现道路 |
| 依据 | EXP_P025 · EXP_P027 · EXP015 |
