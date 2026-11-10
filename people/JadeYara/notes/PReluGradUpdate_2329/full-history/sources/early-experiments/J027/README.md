

## J027 — 固定J026 tiling，缓存低精度权重的FP32转换

目标限定为13us大输入的代价核算与一个可执行候选。J026 split-C仍每batch Cast同一权重，改为首batch转换到已有neg缓冲并驻留；Compare读取features完成后，复用featuresFloat保存weight*grad（沿用J008已验证的生命周期思路）。Host、tile、核数、UB分配不变，FP32数学路径不变。不重复J007双缓冲扫描。上游hicann/ops-nn b5f577d6 CopyWeightTile/ProcessChannelNcSplitCWeightReuse将BF16权重转换移出batch循环，许可与原文已归档J021；本次只借鉴思路。另设J027IO诊断版本，仅保留同tiling搬运/队列并输出grads/features，用于观察相同结构搬运成本，不是合法算子候选或严格硬件下界。先708原精度，再63数据集三dtype四轮J026/J027；IO仅3形状×3dtype。主形状两种msprof op replay模式核实L2影响。全部保留，不提交/push。


本轮已完成，实测结论、IO下界条件及归档见 [RESULTS.md](RESULTS.md)。J027IO仅诊断，不是合法算子实现。
