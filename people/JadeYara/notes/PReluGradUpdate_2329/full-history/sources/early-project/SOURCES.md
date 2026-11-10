# J001 开源参考与许可记录

检索日期：2026-11-03。固定参考：GitHub可读副本 `hicann/ops-nn@07b761da39bcddaaa834408e9eb0127c454fe07f`。官方 GitCode 相关页面限流，未确认两端当前逐文件一致。

根 [LICENSE](https://github.com/hicann/ops-nn/blob/07b761da39bcddaaa834408e9eb0127c454fe07f/LICENSE) 和参考源码文件头为 **CANN Open Software License Agreement Version 2.0**，不是Apache/MIT。该许可面向Huawei AI处理器/CANN软件用途，分发源码衍生物需保留声明并附许可；未来若直接移植源码，应随代码保留这些材料。本候选只借鉴调度思路，没有复制参考仓源码。

## 实际阅读并采用的思路

- [910B Prelu 注册](https://github.com/hicann/ops-nn/blob/07b761da39bcddaaa834408e9eb0127c454fe07f/experimental/activation/prelu/op_host/prelu_def.cpp)：确认其支持ascend910b。
- [Prelu Host](https://github.com/hicann/ops-nn/blob/07b761da39bcddaaa834408e9eb0127c454fe07f/experimental/activation/prelu/op_host/prelu_tiling.cpp#L111)：weightSize!=1才进通道模式。借鉴此分派思路，使本题C=1、weights=[1]保持标量广播。没有复制前向Prelu算术或固定核数常量。

## 已研究，尚未合入

- [InitScalar](https://github.com/hicann/ops-nn/blob/07b761da39bcddaaa834408e9eb0127c454fe07f/experimental/activation/prelu/op_kernel/prelu.h#L239)：每核读取一次共享权重，后续tile复用。可独立试验，当前J001未改Kernel。
- [NC resident weights](https://github.com/hicann/ops-nn/blob/07b761da39bcddaaa834408e9eb0127c454fe07f/experimental/activation/prelu/op_kernel/prelu.h#L329)、[构建权重向量](https://github.com/hicann/ops-nn/blob/07b761da39bcddaaa834408e9eb0127c454fe07f/experimental/activation/prelu/op_kernel/prelu.h#L614)、[split-C](https://github.com/hicann/ops-nn/blob/07b761da39bcddaaa834408e9eb0127c454fe07f/experimental/activation/prelu/op_kernel/prelu.h#L818)：S=1时批量向量化、权重驻留及跨batch复用；本题有额外输入/输出/比较缓冲，需重算UB预算和核分配，不能整套套用。
- [同名PReluGradUpdate arch35 DAG](https://github.com/hicann/ops-nn/blob/07b761da39bcddaaa834408e9eb0127c454fe07f/activation/p_relu_grad_update/op_kernel/arch35/prelu_grad_update_dag.h)：三输入广播、乘法/Compare/Select、两输出，无reduce；使用新架构RegTensor/__VEC_SCOPE__/BroadcastSch，不能直接替换910B工程。
- [PReluGrad融合拆分](https://github.com/hicann/ops-nn/blob/07b761da39bcddaaa834408e9eb0127c454fe07f/activation/p_relu_grad_update/op_graph/fusion_pass/prelu_grad_fusion_pass.cpp)：Update与Reduce分离。因此本题保持update与输入同shape，不引入权重梯度归约。

未采纳：前向Maxs/Mins/Add公式、未经910B验证的BF16 Cast、新架构专用API、P102已撤回的DataCopy微调。P103基线属于团队协作仓，复制到本人的seat遵循原仓README与JadeYara开工指引。
