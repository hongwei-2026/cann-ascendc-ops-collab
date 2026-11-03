# 910B / CANN 9.1：bf16 不要 Cast

云编报 `not support bf16 type cast` → FATBIN 缺 `.o`。  
bf16↔fp32 用位扩展 / RNE 截断；Compare/Select/Mul 仍在 fp32 LocalTensor。  
`if constexpr` 藏 Cast **不够**（EXP_P023）；头文件里残留的 `bfloat16_t` 辅助函数也会进后端。  
样例为 fp32 时：默认 `PRELU_ENABLE_FP16=0` + `PRELU_ENABLE_BF16=0`，只编 `DT_FLOAT`（EXP_P024 → P025）。  
见 `EXP002` / `EXP007` / `EXP010` / `EXP011`。
