# 状态 — LayerNormV3_2326

| 项 | 内容 |
|----|------|
| 状态 | **`实现中`**（语义自测已过；NPU `-e` 未跑） |
| Owner | yuhongming-2026 |
| 更新日期 | 2026-11-06 |
| 代码座位 | `people/yuhongming-2026/LayerNormV3_2326/workdir/code/` |
| 本地闸 | [EXP_L001](explorations/EXP_L001-semantic-pass.md)：`local_semantic_check.py` **ALL PASSED** |
| AI 记录 | [SESSION](explorations/AI-SESSION-20261106-card.md) · [DIALOG](explorations/AI-DIALOG-TRANSCRIPT-20261106.md) |
| 下一闸 | 910B 上 `bash build.sh -e`，再视泛化/性能日志决定要不要改 tile |

## 一句话

LayerNormV3 已按模板落地到 yuhongming 座位：float32 两遍归约，fp16/bf16 只做 IO 转换。无 NPU，不能把设备精度写成通过。
