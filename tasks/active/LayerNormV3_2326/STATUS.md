# 状态 — LayerNormV3_2326

| 项 | 内容 |
|----|------|
| 状态 | **`实现中`**（910B2 样例和一组泛化形状已通过；平台隐藏用例还没在 Judge 上复测） |
| Owner | yuhongming-2026 |
| 更新日期 | 2026-11-08 |
| 代码座位 | `people/yuhongming-2026/LayerNormV3_2326/workdir/code/` |
| 本地闸 | [EXP_L001](explorations/EXP_L001-semantic-pass.md)：`local_semantic_check.py` **ALL PASSED** |
| 设备闸 | 2026-11-08，910B2，CANN 9.1.0。样例 `build.sh -e`：y 0/192，mean 0/4，rstd 0/4。另在设备上跑了非对齐列、多核、5000 列、fp16/bf16、x 为 half/bf16 且 gamma 为 float，全部 0 mismatch |
| AI 记录 | [SESSION](explorations/AI-SESSION-20261106-card.md) · [DIALOG](explorations/AI-DIALOG-TRANSCRIPT-20261106.md) |
| 下一闸 | 平台 Judge 隐藏用例 |

## 一句话

LayerNormV3 已按模板落地到 yuhongming 座位。910B2 上样例和非对齐、多核、半精度、gamma 为 float 的自测都已通过。平台隐藏用例还没复测。
