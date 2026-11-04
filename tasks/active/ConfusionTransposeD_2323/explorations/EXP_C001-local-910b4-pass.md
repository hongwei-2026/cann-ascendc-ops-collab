# EXP_C001 — 本地 910B4 样例 -e PASS

| 字段 | 内容 |
|------|------|
| 题目 | ConfusionTransposeD_2323 |
| 日期 | 2026-11-04 |
| 作者 / AI | yuhongming-2026 · Cursor |
| 阶段 | 功能 / 精度（样例） |
| 结论 | **采纳** |
| 座位 | `people/yuhongming-2026/ConfusionTransposeD_2323/workdir/code/` |

## 结果

- `bash build.sh -e` → `PASS: y.bin, mismatched=0/120` → **PASSED**
- 方案：UB 常驻整输入；按 `transpose_first` 做输出下标→输入下标映射后位拷贝

## 假设边界

超 UB 的大 tensor 当前 tiling 直接失败；后续若需泛化再开分块路径。
