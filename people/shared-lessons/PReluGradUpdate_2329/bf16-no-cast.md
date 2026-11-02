# 910B / CANN 9.1：bf16 不要 Cast

云编报 `not support bf16 type cast` → FATBIN 缺 `.o`。  
bf16↔fp32 用位扩展 / RNE 截断；Compare/Select/Mul 仍在 fp32 LocalTensor。  
见 `explorations/EXP002-bf16-cast-backend-fail.md`。
