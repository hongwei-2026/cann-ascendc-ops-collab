# EXP010 — P023 仍 bf16 Cast；P024 关 bf16 key

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-03 |
| 日志 | `explorations/logs/EXP_P023_20261103_123326/` |
| 关联 | EXP002/007 · P020/P023 |

## 现象

```text
fatal error: error in backend: not support bf16 type cast
```

三条 tiling key（含 fp32/fp16）同炸 → FATBIN 缺 `.o`。P023 源码里虽有 `if constexpr` 位转换，仍不够。

## 处置（P024）

1. `DtypeBridge` **全特化**：Cast 只出现在 `half` 特化里  
2. **`PRELU_ENABLE_BF16=0`**：TPL / OpDef / tiling / entry 默认不编 bf16  
3. 样例为 fp32，先验证 Compare(+zero)+count Select  

重开 bf16：三处把头里的 `PRELU_ENABLE_BF16` 改为 `1`（或统一 `-DPRELU_ENABLE_BF16=1`）。
