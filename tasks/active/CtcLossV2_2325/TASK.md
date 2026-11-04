# 题目：CtcLossV2

## 元信息

| 项 | 内容 |
|----|------|
| 题目 ID | 2325 |
| 类型 | A 算子 |
| 硬件 | Ascend 910B |
| CANN | 9.1.x |
| 对标 | ops_legacy TIK CTCLossV2（非标准 PyTorch CTC） |
| 模板 | `CtcLossV2_problem_2325_template.zip` |

## 验收摘要

- 功能对齐 TBE/TIK；精度 float 1e-4
- 样例：`bash build.sh -e`

## 非目标

- 本阶段不追超大 S/T 性能；先通样例与 TIK 语义
