# 题目分类索引

## 状态说明

| 标签 | 含义 |
|------|------|
| active | 进行中，台账持续更新 |
| archive | 已结束或长期挂起；保留教训，默认不再投入 |
| 规范就绪 | 仅规范/任务卡，未开发 |
| 未验收 | 曾推进但未作为通过交付 |

## Active

| 题目 | 目录 | 类型 | 合入仓（预期） | 状态 |
|------|------|------|----------------|------|
| ConfusionTransposeD (2323) | [active/ConfusionTransposeD_2323](active/ConfusionTransposeD_2323) | A 算子 | CANNJudge / 本仓座位 | 实现中 |
| CtcLossV2 (2325) | [active/CtcLossV2_2325](active/CtcLossV2_2325) | A 算子 | CANNJudge / 本仓座位 | 实现中 |
| LayerNormV3 (2326) | [active/LayerNormV3_2326](active/LayerNormV3_2326) | A 算子 | ict / 开源仓算子工程 | 实现中（语义自测过，NPU `-e` 未跑） |
| PadV3Grad (2328) | [active/PadV3Grad_2328](active/PadV3Grad_2328) | A 算子 | ict / 开源仓算子工程 | 实现中（语义自测过，NPU `-e` 未跑） |
| PReluGradUpdate (2329) | [active/PReluGradUpdate_2329](active/PReluGradUpdate_2329) | A 算子 | ict / 开源仓算子工程（以任务书与模板为准） | 分析定稿（SUMMARY 完成） |

## Archive

| 题目 | 目录 | 类型 | 备注 |
|------|------|------|------|
| SpMV (08-SPMV) | [archive/08-SPMV](archive/08-SPMV) | A 算子 | 未最终验收；设计曾合入 |
| Basic API CUBE 指针化 | [archive/09-Basic-API-CUBE](archive/09-Basic-API-CUBE) | B API | 未最终验收；hidevlab 教训已进 skill |
| SageAttention2 | [archive/SageAttention2](archive/SageAttention2) | A 算子 | 任务书归档；本仓暂不开发 |

## 新建题目

1. 复制 `docs/templates/task-scaffold/` → `tasks/active/<ID>_<短名>/`  
2. 填 `TASK.md` / `STATUS.md` / 首行 `path-ledger.md`  
3. 更新本索引表  
4. PR 合入 `main`
