# 910B / CANN 9.1：bf16 不要 Cast

云编报 `not support bf16 type cast` → FATBIN 缺 `.o`。  
bf16↔fp32 用位扩展 / RNE 截断；Compare/Select/Mul 仍在 fp32 LocalTensor。  
`if constexpr` 藏 Cast **不够**（EXP_P023）；用 `DtypeBridge` 全特化，Cast 只留在 `half`。  
暂时编不过时：`PRELU_ENABLE_BF16=0` 先只编 fp16/fp32（样例为 fp32）。  
见 `EXP002` / `EXP007` / `EXP010`。
