# 910B / CANN 9.1：bf16 不要 Cast

云编报 `not support bf16 type cast` → FATBIN 缺 `.o`。
bf16↔fp32 用位扩展 / RNE 截断；Compare/Select/Mul 仍在 fp32 LocalTensor。
`if constexpr` 藏 Cast **不够**（EXP_P023）；头文件里残留的 `bfloat16_t` 辅助函数也会进后端。

## P029 加严（EXP015/016）

`DtypeBridge` **显式特化**里的 `ToFloat`/`Cast`/`static_cast<half>` 会编进**每一个** tiling key 的 TU，连纯 fp32 key 也炸。  
做法：删显式特化；`float`/`half` 原生 Compare+Select；bf16 位转换只写在 `if constexpr (is_same<bfloat16_t>)` 分支内。
样例为 fp32 时：默认 `PRELU_ENABLE_FP16=0` + `PRELU_ENABLE_BF16=0`，只编 `DT_FLOAT`（EXP_P024 → P025）。  
见 `EXP002` / `EXP007` / `EXP010` / `EXP011`。
