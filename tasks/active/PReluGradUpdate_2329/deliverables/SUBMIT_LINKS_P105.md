# P105 本地候选入口 — 提前准备标量权重

2026-11-04；席位 hongwei-2026；基于 P103 `47db93e`。只把原权重准备块移到 Compute 入口、DeQue 之前，Host/tiling/缓冲/数值运算/同步不变。P104 不采纳。

**状态：本地三组检查通过；CANN 三 key 编译、设备精度与平台计时待测。没有 P105 提速或得分结论。**

Kernel SHA256（UTF-8/LF）：`4f100a73615ba10f0416f3cccce8b58eb474b101e01eaafde608f8ce9363520c`

## 实时源码

代码根目录：`D:/sageAttention2 算子开发任务书/people/hongwei-2026/PReluGradUpdate_2329/workdir/code`

- [op_host/p_relu_grad_update_def.cpp](<D:/sageAttention2 算子开发任务书/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_host/p_relu_grad_update_def.cpp>)
- [op_host/p_relu_grad_update_infershape.cpp](<D:/sageAttention2 算子开发任务书/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_host/p_relu_grad_update_infershape.cpp>)
- [op_host/p_relu_grad_update_tiling.cpp](<D:/sageAttention2 算子开发任务书/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_host/p_relu_grad_update_tiling.cpp>)
- [op_kernel/p_relu_grad_update.cpp](<D:/sageAttention2 算子开发任务书/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update.cpp>)
- [op_kernel/p_relu_grad_update.h](<D:/sageAttention2 算子开发任务书/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update.h>)
- [op_kernel/p_relu_grad_update_tiling_data.h](<D:/sageAttention2 算子开发任务书/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update_tiling_data.h>)
- [op_kernel/p_relu_grad_update_tiling_key.h](<D:/sageAttention2 算子开发任务书/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update_tiling_key.h>)

以上为实时工作区链接，后续候选会改变文件内容；交测前核对指纹。P103 冻结源码在历史提交 `47db93e`；历史平台 42.89 分不能当作本候选成绩。

## 复核与交测

在代码根目录运行：

```powershell
python -B tests/weight_schedule_check.py
python -B tests/compute_schedule_check.py
python -B tests/local_semantic_check.py
```

随后按现有云测流程编译三 key、验证设备精度；SOURCE 核对源码/构建指纹一致，归档日志后平台计时。无收益回 P103，详见 EXP025。仅本地修改，不提交、不推送。
