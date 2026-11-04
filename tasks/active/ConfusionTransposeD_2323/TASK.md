# 题目：ConfusionTransposeD

## 元信息

| 项 | 内容 |
|----|------|
| 题目 ID / problem | 2323 |
| 类型 | A 算子 |
| 合入仓 | cann-ascendc-ops-collab / CANNJudge |
| 硬件 | Ascend 910B |
| CANN | 9.1.x |
| 对标实现 | TBE ConfusionTransposeD；`npu_confusion_transpose` |
| 模板包 | `ConfusionTransposeD_problem_2323_template.zip` → `materials/` |

## 验收摘要

### 功能

- 对齐原 TBE：`transpose_first` 控制 permute↔reshape 顺序
- attrs：`perm`、`shape`、`transpose_first`
- 支持模板声明的多 dtype / ND

### 精度

- float `1e-4` · half `1e-3` · int `0`（CANNJudge 默认）

### 性能

- 先通样例功能；性能后续有界优化

### 测试

- `bash build.sh -e` 本地 910B 样例对拍
- UT：`bash build.sh -u`

## 非目标

- 本阶段不做超大 tensor 超 UB 的通用分块（tiling 对超 UB 直接失败）
