# 状态 — PadV3Grad_2328

| 项 | 内容 |
|----|------|
| 状态 | **`实现中`**（语义自测已过；NPU `-e` 未跑） |
| Owner | yuhongming-2026 |
| 更新日期 | 2026-11-07 |
| 代码座位 | `people/yuhongming-2026/PadV3Grad_2328/workdir/code/` |
| 本地闸 | [EXP_P001](explorations/EXP_P001-semantic-pass.md)：`local_semantic_check.py` **PASS** |
| AI 记录 | [SESSION](explorations/AI-SESSION-20261107-card.md) · [DIALOG](explorations/AI-DIALOG-TRANSCRIPT-20261107.md) |
| 下一闸 | 910B 上 `bash build.sh -e` |

## 一句话

PadV3Grad 已按模板 reference 落到 yuhongming 座位。constant 按行分核拷贝，其它模式单核 float32 折回。无 NPU，不能把设备精度写成通过。
