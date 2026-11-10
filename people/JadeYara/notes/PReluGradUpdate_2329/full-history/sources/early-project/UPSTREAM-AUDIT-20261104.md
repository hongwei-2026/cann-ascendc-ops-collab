# 仓库建议与 J013–J018 实现对照（2026-11-04 阶段报告）

结论：确实参考了仓库，但“整核权重驻留”此前没有完整落实；不能将“读过方案”写成“全部采用”。此外，之前固定参考的 22ec6d6/P103 已不是远端最新 tip；本轮重新查询 GitHub main 得到 e62693c8bd5946eebcf2ffaacb37d2bcb892bbbf，当前试验 P105 的实现提交为 2e378eb557afd7bb346dee1e61e0d5b9ba702960。P104已拒绝，P105只有原仓样例正确性、无已确认平台收益。本轮已补做 P105 思路的同机实验 J017，未发现可采用收益。

## 逐项落实情况

| 原始建议 / 证据 | J013 或后续实际实现 | 结论 / 未采用理由 |
|---|---|---|
| P103 Compute：update Select 先读零，dx 后覆盖；6操作/4屏障 | J003/J013沿用该依赖；J008复用死去的features缓冲，仍保留相应RAW/WAR屏障 | 已采用核心依赖，不把少屏障等同加速 |
| AI-precision-P103-validation §7 第1项：weightLen=1连续块，C=1不截断 | J001 Host修正并合入J013 | 已采用并实测；并非新造语义 |
| 同文件§7 第2项：S=1加载权重向量、Mul、明确跨batch周期 | J004→J013 S=1路径 | 已采用；但仍按C边界重复读权重，只有部分落实 |
| 同文件§7 第4项：按profile决定BF16转换 | J003原生Cast；在910C/CANN9.0单独编译验证 | 已采用；600k BF16由J001约673us降至约9us，不能把旧910B Cast失败机械套用到当前设备 |
| SUMMARY“模式R权重向量整核只展开一次”；design_final §3.3.3 L171–179 | J013未实现；J015仅每tile一个周期GM读取+UB展开 | 明确缺口；已补J016，发现边界错误，修正版J018准备好但尚未执行 |
| SOURCES.md 上游 Prelu InitChannelNcResidentWeightReuse、BuildNcResidentWeightVec、split-C | 上游会在Init加载权重并构造驻留向量；J015只实现tile内复用 | 仅借鉴思路，未复制第三方源码；整核复用此前未落实，split-C/通用padding布局未直接移植 |
| 小inner多通道合并 / 模式R、模式C | J013普通通道仍逐通道Muls；只对inner=1做专门路径 | 尚未实现通用多通道合并。当前无证据证明完整移植更快，应按shape/profile单独验证 |
| 双缓冲/预取 | J007真实96精度、两轮A/B | 无稳定收益：600k FP32两轮慢0.30/0.73us；低精度方向不一致，未采用 |
| 共享中间缓冲 | J008→J013复用featuresFloat、UB系数21/29 | 已采用并记录实际精度/性能，不与独立零缓冲混同 |
| P102对齐搬运DataCopy分支 | 保留DataCopyPad | 上游DEC001已回退，不重复 |
| 最新P104独立常驻零缓冲 | 未合入 | 上游DEC002：前三公开点较P103历史图慢约10–14%，不重复该已拒绝路线 |
| 最新P105权重读取/解码前移到DeQue前 | J017基于J013仅搬移原标量块，S1/Host/同步不变 | 本轮435精度通过，39输入×2版本×4轮=312组输出全对；没有稳定收益，不采用 |

符号不能混读：旧design_final的 S=C×inner（sampleSize）；本轮所称“S=1路径”指inner=1。文档描述的Mode R不是对任何C都自动适用。原设计还有旧缓冲数/例子系数残留，不能不核实际InitBuffer就整段复制。

## 为什么大通道仍慢：具体证据

固定P103源码的Process在每次循环把currentNum截到innerSize剩余长度；inner=1时可退化为每元素一次tile。J004/J013将同一C段连续向量化，已将若干本地大通道输入从约300us降至约9–12us，但跨batch仍重复读相同权重。

J013/J014/J015同机同输入四轮消融：BF16 [1001,512,1] 的Task平均分别43.567/43.144/14.135us，MTE2活跃占比76.80%/80.64%/59.32%。J014合并Compute但仍逐段读GM，收益不稳；J015只再改变权重复用方式就明显缩短延迟。代码差异加A/B支持重复GM权重读取是该路径重要成本。各pipeline可重叠，比例不能相加，剩余14us也不能按比例直接归因为某一条指令。

FP32 [127,256,1] 平均9.677→6.390us；三dtype该shape平均改善约28–34%。[1001,512,1]约68–75%。这些仍不是4–5us全点达标。J015在未触发优化的极小S1上有约0.045–0.175us位置复核平均开销，所以J013保留综合基线，J015仍是专项候选。

600k普通路径与“大通道权重”不是同一路径：本轮输入[600000,1,1]的wlen=1，不搬C权重向量。J013/J015约9us，MTE2约45%、Scalar约36%、Vector约25%；J015不改此kernel。Host的80%UB预算让FP32每核约15k元素分3个tile，预算/循环是下一可测项，尚未实施，不能宣称4us已可达。

## 真实补测，而非仅补说明

J016补整核驻留：既有399输入全对，新36输入32通过、4失败；失败仅FP16/BF16 [127,4096,1]和[129,4096,1]的dx，update全对。Host将tileLen对齐成C后，mergeWeights旧条件tileLen>C变false，Process继续在C边界截断，破坏缓存权重相位。J018仅修正“驻留时也禁用通道截断”的条件，源码另存；尚未编译/运行。J016没有跑性能，不推测驻留收益。

J017补P105思路：435/435设备精度、312组计时输出通过。普通路径没有输入在四轮中稳定改善≥0.15us。600k FP32/FP16/BF16平均变化为+0.020/+0.065/+0.098us；方向随顺序变化，未观察到可采用收益。S=1代码没有改动，其控制组也有零点几us波动，不能归因给P105。J017不是原仓完整P105包，而是该单项调度改动在J013之上的隔离实验。

## 口径与来源

P103的42.89及7.36/7.30/7.24/5.83/6.07us是历史平台五点；输入shape、dtype和原始逐次记录未知。当前实验为Ascend910_9362/CANN9.0的msprof kernel Task Duration，每输入25次、丢前5次。上游新P104日志包含CANN9.1路径。环境、输入和计时边界没有绑定，不能用本地600k的9us去直接对比平台的6us或TBest约4us。没有声称与TBest同输入对比。

- [P103固定源码](https://github.com/hongwei-2026/cann-ascendc-ops-collab/blob/22ec6d63da92ef2193dd66d1666f5d238cf3d08d/people/hongwei-2026/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update.h)
- [最新STATUS](https://github.com/hongwei-2026/cann-ascendc-ops-collab/blob/e62693c8bd5946eebcf2ffaacb37d2bcb892bbbf/tasks/active/PReluGradUpdate_2329/STATUS.md)
- [P105实验](https://github.com/hongwei-2026/cann-ascendc-ops-collab/blob/e62693c8bd5946eebcf2ffaacb37d2bcb892bbbf/tasks/active/PReluGradUpdate_2329/explorations/EXP025-early-weight-preparation.md)
- [P104回退决定](https://github.com/hongwei-2026/cann-ascendc-ops-collab/blob/e62693c8bd5946eebcf2ffaacb37d2bcb892bbbf/tasks/active/PReluGradUpdate_2329/decisions/DEC002-p104-platform-regression.md)
- [Prelu整核驻留参考](https://github.com/hicann/ops-nn/blob/07b761da39bcddaaa834408e9eb0127c454fe07f/experimental/activation/prelu/op_kernel/prelu.h#L329)
- [J014/J015完整本地结果](experiments/J015/RESULTS.md)

当前断点：J014/J015完整主/补证据已本地归档校验。J016失败及J017完成结果仍需下载完整云端归档；本地已有状态/诊断日志及J017全部摘要。J018源码已准备，未启动。没有在途设备作业、未提交/未推送。按用户要求先交本报告，不把待执行项写成已完成。


## J018 完成：驻留相位修复与完整归档（2026-11-04）

J018 单条件修复构建通过，6个失败边界的三dtype复核全部正确，435既有+27新增=462独立精度全部通过。J016同输入复现原4失败，新增相位再发现4失败，8项排除性能。J013/J015/J018各69输入、J016合格61输入，四轮共1072组/26800 kernel记录，输出全通过。Host规则计算28个驻留输入平均全部改善，27个每轮至少快0.15us；BF16 [1201,512,1] 16.384→10.720us，[127,4096,1] 10.880→9.517us。非驻留路径回退：BF16 [2001,512,1] 23.119→25.679us，小S=1也退化，故J018仅保留专项候选，不替换J013/J015。下一步候选是独立编译隔离驻留路径，本轮未实施。

J016/J017/J018完整云端证据已下载，SHA/CRC/全部二进制索引验证通过：J016-J018-910C-evidence.zip 354043635 bytes，SHA 2187aaced95a13649de82f014af9b5f0f672eb18cb92d1edccaf1eaf83b507ef。报告 people/JadeYara/PReluGradUpdate_2329/experiments/J018/RESULTS.md；归档 tasks/active/PReluGradUpdate_2329/explorations/logs/EXP_J016_J018_RESIDENT_910C_20261104。无在途NPU作业，无提交或push。


## J019/J020 独立编译实验完成

两版各498精度通过，1680+780+420=2880组输出全通过。完整对照见 people/JadeYara/PReluGradUpdate_2329/experiments/J020/RESULTS.md。

## 本轮结论与采用状态

**独立分派实现与精度通过，但尚未全面隔离小输入回退；J019/J020 不作为新的全量默认候选。** J013 保持综合基线，J015 与 J018 的原专项候选继续保留。本轮没有声称达到未知榜单的 4–5 μs 目标。

固定位置复核给出了更具体的边界：

- 大输入之前的 [3,5,1]，J020 相对 J013 的平均差值为 FP32 +0.028、FP16 +0.005、BF16 -0.025 μs，接近原水平。
- 驻留大输入之后，同一小输入 FP16 为 2.950→3.550 μs，BF16 为 2.970→3.592 μs；四轮差值分别约 +0.57～+0.63、+0.59～+0.64 μs。
- 非驻留大输入之后，FP16 小输入 2.948→3.347 μs，BF16 2.977→3.362 μs；四轮分别 +0.34～+0.43、+0.37～+0.40 μs。J019 也有相同现象。因此这不是只在长序列里出现的一次异常。
- 大输入的权重驻留收益保留：固定位置 BF16 [1201,512,1] 为 J015 17.637、J018 10.950、J019 10.802、J020 10.732 μs；FP16 同 shape 为 18.247、10.775、11.220、10.942 μs。新分派没有证明所有 dtype 都优于 J018。
- 非驻留 [2001,512,1] 的固定位置 J020 与 J015 相比：FP32 +0.056、FP16 -0.638、BF16 -0.185 μs；原 J018 大幅回退在此短批没有复现，但不能据此抹去早期测试差异。
- 600k 普通路径 J020 相对 J013 的固定位置平均差约 +0.075/+0.000/-0.045 μs（FP32/FP16/BF16），没有新算法收益。

编译期常量化不是普遍收益：J019 在 [40001,16,1] 的低精度路径有额外改善，但其他驻留路径的差值随 campaign 变化，J020 恢复原实现后也没有解决所有小输入后继开销。当前证据不能单独把问题归因于指令缓存、频率或某个同步事件。

固定位置复核控制了输入顺序和位置，但前序大输入在各版本中的实际执行时长不同（例如 J013 数十 μs、优化版十余 μs），没有把时钟、缓存或前序执行状态固定成相同值。因此低精度小输入的差异是该调用序列下的实测现象，不足以证明小输入核代码本身增加了同等开销。若要求跨序列都无回退，此验收条件尚未满足。

进一步读取已有 profiling：BF16 tiny_after_resident 的 J013→J020，Task Duration 为 2.970→3.592 μs，aiv_time 为 2.434→3.066 μs；FP16 对应为 Task 2.950→3.550 μs、aiv_time 2.413→2.998 μs。额外差异主要伴随设备 AIV 执行区间增长，下一步应从该区间查起，不能将其归于 SSH 或本机网络延迟。Scalar/Vector/MTE2 活跃比例仍相互重叠，不足以定位独占等待原因。原始聚合见 position-metrics.json。

后续应先定位“低精度小输入紧跟大输入”的稳定额外开销，结合入口反汇编、代码布局与流水线事件做一个有针对性的实验；不继续同时修改分派和数值路径。本轮没有进一步修改代码。


归档 tasks/active/PReluGradUpdate_2329/explorations/logs/EXP_J019_J020_DISPATCH_910C_20261104；增量归档与指定J016-J018完整归档共同校验通过，可恢复全部证据。无在途NPU作业、无提交或push。
