# EXP012 — P025 样例精度 PASS（fp32-only）

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-03 |
| 日志 | `explorations/logs/EXP_P025_20261103_125446/` |
| tip | `a6f7b6a` |

## 结果

```text
[INFO] Build completed successfully!
PASS: dx.bin, mismatched=0/192
PASS: update.bin, mismatched=0/192
PASSED
[INFO] Example completed successfully!
```

## 含义

1. **编译门**已过：默认仅 fp32，避开 910B bf16/half Cast 后端坑。  
2. **样例功能/精度**过：`Compare(+zero)` + count `Select` + 独立 neg/fg buffer 在 seed 样例上正确。  
3. 下一步：恢复 fp16/bf16（仍禁止 bf16 Cast），或先用本包平台试交。
