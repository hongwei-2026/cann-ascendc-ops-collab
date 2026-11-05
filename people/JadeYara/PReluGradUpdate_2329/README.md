# JadeYara · PReluGradUpdate_2329

- **云端同步用这一版**：[workdir/code/](workdir/code/)，当前为冻结 **J035**，完整 46 个文件。
- 共享台账 / 材料 / EXP：[tasks/active/PReluGradUpdate_2329/](../../../tasks/active/PReluGradUpdate_2329/)；先核对其他作者已经试过的路线、结果和停止原因，再继续优化。
- 共识教训：[people/shared-lessons/PReluGradUpdate_2329/](../../shared-lessons/PReluGradUpdate_2329/)。旧工具链结论需结合当前 910C/CANN 9.3 的证据判断。
- 自己的进展、困难与后续方案：[个人笔记](../notes/PReluGradUpdate_2329/README.md)，原有文档与台账均保留。

## 当前版本与结果

J035 已实现 normal、S1、merged、resident、split-C 路径；大通道 split-C 固定 C 块并跨 batch 复用权重，FP32 使用 UB 相位布局。两个输出均保持输入 shape，本算子不做权重梯度归约。

用户确认官方提交 **1148578** 对应 J035，五点均 Pass，耗时为 **7.15 / 7.29 / 7.11 / 5.88 / 6.06 μs**。对应关系来自用户确认，官方输入 shape、dtype 与环境尚未知。

完整源码及指纹见 [J035 版本说明](../notes/PReluGradUpdate_2329/experiments/J035/README.md)。本次只整理目录和文档，没有修改这 46 个源码文件。`workdir/code/README.md` 与 `FIX_NOTES.md` 保留冻结版本的历史内容；当前状态以本页和个人笔记为准。

## 当前困难与下一步

J048 虽通过精度，官方前三点却比 J035 慢 1.06–1.30 μs；局部缓存收益还不能支持整体替换。J053 已用同进程 A/A/B/C 区分计时漂移和缓存收益：目标低精度本地约快 0.44 μs，高复用约快 1 μs；这不证明官方回退已解决。

后续先对照公开仓库的试错记录，再评估尚未排除的有限分派方案。已失败的全面双缓冲、每 tile 重读权重等路线，不因换代理重新开始。详见 [难点与方案](../notes/PReluGradUpdate_2329/optimization-status.md)、[其他作者实际源码对照](../notes/PReluGradUpdate_2329/source-comparison.md)、[J053 结果](../notes/PReluGradUpdate_2329/J053-device-validation.md)。本次不新增实验或测试。

CodeArts 同步范围仅为自己的 `people/JadeYara/PReluGradUpdate_2329/workdir/code/`。不要复制或覆盖 hongwei 的工程；官方提交由用户本人完成。
