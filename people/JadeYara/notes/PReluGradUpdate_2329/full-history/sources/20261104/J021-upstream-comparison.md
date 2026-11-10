# 上游权重复用对照（2026-11-04）

核查版本：[hicann/ops-nn b5f577d6bfdbc543d42638ef8635e3f5ef8cb0c8](https://github.com/hicann/ops-nn/commit/b5f577d6bfdbc543d42638ef8635e3f5ef8cb0c8)。根 LICENSE 及文件头均为 CANN Open Software License Agreement Version 2.0。这里只读比较设计，未移植代码。

`experimental/activation/prelu/op_host/prelu_def.cpp` 注册 `ascend910b`，计算的是单输出 PReLU；本机为 Ascend910_9362、目标 ascend910_93（dav-c220-vec），本题是两输出 PReluGradUpdate。注册、内存流量和功能并不相同。

`prelu.h::ProcessChannelNcSplitCWeightReuse` 按 `cTileIdx=taskIdx/batchSize`、`nIdx=taskIdx%batchSize` 遍历。只在 `cTileIdx != lastCTileIdx` 时 `CopyWeightTile` 和 `BuildNcWeightVec(1)`，然后遍历同一通道块的 batch。这样可在完整 C 不能驻留时摊销局部通道块权重搬运。

当前 J018/J020 的 resident 路径：保持平坦连续数据分块；CopyIn 在 progress==0 才搬权重，Compute 后续复用 residentWeightsLocal；只在 Host 能把 tile 对齐到完整 C 且不增加循环数时启用。因此已经实现符合条件时的整核复用，但**没有采用 split-C 的遍历重排**。不符合驻留条件的大 C 或循环数会增加的形状，仍是 merged/legacy 路径；不能将已有驻留优化声称为已覆盖 split-C。

split-C 仍可作为下一项独立实验：先核对应形状权重流量与循环/地址开销，注意跨 batch 的输入输出搬运可能更碎。它不能解释本轮同一个 tiny kernel 在不同先行工作负载后的 ICache miss；本轮 J021/J022 因而先完成该已复现回退的定位与最小改动。

源码依据：
- [PReLU kernel](https://github.com/hicann/ops-nn/blob/b5f577d6bfdbc543d42638ef8635e3f5ef8cb0c8/experimental/activation/prelu/op_kernel/prelu.h)
- [注册](https://github.com/hicann/ops-nn/blob/b5f577d6bfdbc543d42638ef8635e3f5ef8cb0c8/experimental/activation/prelu/op_host/prelu_def.cpp)
- [LICENSE](https://github.com/hicann/ops-nn/blob/b5f577d6bfdbc543d42638ef8635e3f5ef8cb0c8/LICENSE)

production PReluGradUpdate arch35 的 950/350/960dt 注册信息由上游研究提供；本轮未重新逐行核验该路径，不作为 910C 兼容证明。
