# 平台粘贴用本地超链接（P105 · tip `2e378eb`）

2026-11-04：云测 EXP_P105_204700 **0/192 PASS**（三 key Done；kernel LF 一致）。相对 P103 仅前移权重准备；平台计时待测。对照基线：P103 历史 7.36/7.30/7.24/5.83/6.07μs / 42.89。无 ≥2% 同机收益 → 回 P103。

Kernel SHA256（UTF-8/LF）：`4f100a73615ba10f0416f3cccce8b58eb474b101e01eaafde608f8ce9363520c`

## Host

- [op_host/p_relu_grad_update_def.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_host/p_relu_grad_update_def.cpp)
- [op_host/p_relu_grad_update_infershape.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_host/p_relu_grad_update_infershape.cpp)
- [op_host/p_relu_grad_update_tiling.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_host/p_relu_grad_update_tiling.cpp)

## Kernel

- [op_kernel/p_relu_grad_update.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update.cpp)
- [op_kernel/p_relu_grad_update.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update.h)
- [op_kernel/p_relu_grad_update_tiling_data.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update_tiling_data.h)
- [op_kernel/p_relu_grad_update_tiling_key.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update_tiling_key.h)
