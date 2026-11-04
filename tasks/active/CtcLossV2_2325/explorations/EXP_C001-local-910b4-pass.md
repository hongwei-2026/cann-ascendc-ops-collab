# EXP_C001 — 本地 910B4 样例 -e PASS

| 字段 | 内容 |
|------|------|
| 题目 | CtcLossV2_2325 |
| 日期 | 2026-11-04 |
| 作者 / AI | yuhongming-2026 · Cursor Composer |
| 阶段 | 功能 / 精度（样例） |
| 结论 | **采纳** |
| 座位 | `people/yuhongming-2026/CtcLossV2_2325/workdir/code/` |

## 结果

- `bash build.sh -e` →  
  - `PASS: neg_log_likelihood.bin, mismatched=0/1`  
  - `PASS: log_alpha.bin, mismatched=0/170`  
  - **PASSED**

## 方案要点

- 对齐 `samples/reference.py`（ops_legacy TIK 窗口递推），**不是** arch35 标准 CTC
- Host：`loss[N]`、`log_alpha[N,T,2S+1]`；按 batch 分核
- Kernel：count_trace → 行内 start/end 窗口 → logadd3 → loss_from；UB 常驻 log_probs/targets
