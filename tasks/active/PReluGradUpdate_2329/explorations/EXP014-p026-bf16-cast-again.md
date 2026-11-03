# EXP014 — P026 三 dtype 仍 bf16 Cast

| 字段 | 内容 |
|------|------|
| 日志 | `explorations/logs/EXP_P026_20261103_130902/` |
| tip | `91e1dbf` verify PASS |
| 错误 | `fatal error: error in backend: not support bf16 type cast` |
| 范围 | kernel_0/1/2（fp32/fp16/bf16）全部 FATBIN 失败 |

## 结论

只要编译单元里出现 **bf16 key / `bfloat16_t` 路径**，本 CANN 9.1.0 bisheng 就会报 bf16 Cast（即使代码里没有 `Cast()`）。  
P025 仅 fp32 能编过；P026 开 bf16 复现 EXP002。

## P027

`PRELU_ENABLE_FP16=1`，`PRELU_ENABLE_BF16=0`：先拿 fp32+fp16（half 标量、无 Cast）。bf16 另开道路。
