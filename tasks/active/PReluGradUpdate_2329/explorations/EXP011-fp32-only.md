# EXP011 — P024 仍 bf16 Cast；P025 仅 fp32

| 字段 | 内容 |
|------|------|
| 日志 | explorations/logs/EXP_P024_20261103_124542/ |
| tip | bfb94ad verify PASS |
| 现象 | 仍 not support bf16 type cast；生成 hash=旧 fp16+bf16，无 fp32 hash |
| 根因 | 头文件仍有 bfloat16_t 辅助函数；half Cast 仍编；关 bf16 不够 |
| P025 | PRELU_ENABLE_FP16=0 + BF16=0；只注册/编 DT_FLOAT；Cast/bf16 全进 #if |
