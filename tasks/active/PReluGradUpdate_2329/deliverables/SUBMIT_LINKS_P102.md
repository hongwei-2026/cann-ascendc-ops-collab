# 本地代码入口（当前已恢复 P100；P102 为历史候选）

2026-11-03：P102 未证明性能收益，已撤回。下列链接指向实时工作区，当前内容为 P100，不能作为历史 P102 的源码凭据。

当前 P100 Kernel（LF）SHA256：`ACE056700455F2AF8A2B204DC9690197D9EB9A811831BEEB00295FD6881563A7`；内容对应基线 `3bd65fb`。以下旧 P102 指纹仅保留作历史追溯：

Kernel SHA256（云端/git LF）：`0E58C2B21C0ED1ECC3B8B30340AA144C51D9C63AE5662AC316DD8FAC9787221C`  
（本机 CRLF 工作区可能是 `9E62D7E3…`，同 tip `2e3cbc6`）

## Host

- [op_host/p_relu_grad_update_def.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_host/p_relu_grad_update_def.cpp)
- [op_host/p_relu_grad_update_infershape.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_host/p_relu_grad_update_infershape.cpp)
- [op_host/p_relu_grad_update_tiling.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_host/p_relu_grad_update_tiling.cpp)

## Kernel

- [op_kernel/p_relu_grad_update.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update.cpp)
- [op_kernel/p_relu_grad_update.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update.h)
- [op_kernel/p_relu_grad_update_tiling_data.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update_tiling_data.h)
- [op_kernel/p_relu_grad_update_tiling_key.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update_tiling_key.h)
