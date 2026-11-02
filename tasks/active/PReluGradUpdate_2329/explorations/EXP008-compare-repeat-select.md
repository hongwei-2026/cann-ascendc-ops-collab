# EXP008 — Compare + 显式 repeat Select（P022）

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-02 |
| 作者 | yuhongming-2026 |
| 关联 | EXP004/P015（mask 全 0）· EXP005（Select 别名）· P020（bf16 位转换） |
| 代码 | `people/yuhongming-2026/PReluGradUpdate_2329/workdir/code/` |

## 假设

`CompareScalar` → `Select(count)` 在 910B 样例上未形成有效 bit-mask（恒负支 97/192）。  
按 `design_final`：零向量 + `Compare`；Select 用文档样例的 `mask=64, repeat, BinaryRepeatParams`。

## 改动

1. `Duplicate(dxFloat, 0)` → `Compare(mask, features, dxFloat, GT, cmpCount)`
2. `Select(..., VSEL_TENSOR_TENSOR_MODE, 64, cmpCount/64, {1,1,1,8,8,8})`
3. update 的零向量仍复用 `featuresFloat`（Compare/Mul 之后）
4. 保留 `negFloatBuf_` / `fgFloatBuf_`；bf16 位扩展/RNE

## 待云测

```bash
# CodeArts：只同步 yuhongming workdir
bash people/hongwei-2026/PReluGradUpdate_2329/verify_codearts_source.sh  # 若指向本座位则改路径
DEVICE_ID=0 bash build.sh -j4 -e
```

期望：样例 `[4,6,8]` seed=42 由 WA 95/97 → PASS（或错数远小于 95）。
