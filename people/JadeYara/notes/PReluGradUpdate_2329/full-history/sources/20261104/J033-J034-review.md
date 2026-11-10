# J033/J034：大输入瓶颈复核与一次 UB 布局实验

执行环境：Ascend910_9362，编译目标 ascend910_93 / dav-c220-vec（NPU2201），40 Vector 核，CANN 9.0.0，driver 25.5.5。云端日志日期 2026-11-05（UTC+08）；本地工作区沿用 2026-11-04。所有版本、失败和原始日志保留；没有提交竞赛、push、改网络或安装软件。

## 已经知道什么

当前大输入生产基线仍为 J026。它继承 J024 按通道块遍历 batch、常驻权重的实现，并只在 S=1、逐通道权重、通道32B对齐、C>maxTile、通道块数≤3、N≥2×硬件核数时启用。它并非所有形状的全局最优；J013/J018/J023及专项版本均保留。

在 [127,8192,1] 上，J026 的普通 msprof Task Duration 在不同轮次约为 FP16/BF16 13 μs、FP32 12 μs。历史 J026 对 J023 的同轮目标点改善为低精度约4.6 μs、FP32约1.75 μs。后续版本不能拿不同时间段的单次最小值证明收益。

J030 的实板 MemoryDetail 表明：配对队列 J029 将 Scalar 活跃时间减少约0.30–1.37 μs，但 Vector 指令队列 stall 增加约0.49–1.13 μs，八组配对均同方向；完整 Task 没有稳定收益。MTE2 有较高等待占比，不等于带宽饱和。kernel/application replay 下 L2 命中约7%/99.997%，状态差别真实存在；但工具 main_memory 字节计数未随冷热变化，不能据此当成实际 HBM 读写量或计算严谨的带宽下界。

J031 仿真定位到低精度输入 Cast 完成后仍晚释放队列。J032 将仿真后续 batch 的释放间隔从1036周期压到3周期，但实板18个 dtype/shape 组合四轮没有一组始终更快；FP16目标平均变化仅−0.090 μs，轮间正负翻转。仿真揭示依赖关系，未证明该依赖决定实板关键路径。

因此，不能再把“队列调用多”“晚 FreeTensor”“MTE2 占比高”直接等同于最大可优化耗时；也不能把 Vector/Scalar/MTE 活跃时间相加。这些阶段会重叠，计数模式也会扰动完整 Task。

## 已排除或限制的方向

| 方向 | 已有证据 | 本轮决策 |
|---|---|---|
| 按通道复用权重 | J024/J026 已采用，收益在适用范围复现 | 保留 J026，不重新实现同一机制 |
| 缓存低精度权重的 FP32 转换 | J027 的向量时间降低，但普通四轮有明显回退；固定顺序中 FP32 无改动也变快 | 条件候选保留，不全局推广 |
| tile 不变的输入预取双缓冲 | J028 的8个低精度搬运诊断组合四轮均慢约0.55–0.74 μs | 停止该实现方向 |
| 合并输入/输出队列命令 | J029 Scalar 下降，Vector IQ stall 增长，15组合无稳定四轮收益 | 保留证据，不合入 |
| Cast 后立即释放输入槽 | J032 741精度、1344逐次压力通过，仿真等待缩短，18组合实板无稳定四轮收益 | 不推广 |
| 由单次 profiling 值直接推导4–5 μs | 当前真实输入与榜单输入并不同口径 | 不做该推导 |

## 官方上游和本机工具对照

官方仓为 [cann/ops-nn](https://gitcode.com/cann/ops-nn)。README 要求与 CANN 版本匹配的 tag；本轮未成功取得官方9.0.0 tag的该算子源码，不宣称已经核对该tag实现。

已经核对的固定镜像为 [hicann/ops-nn b5f577d6](https://github.com/hicann/ops-nn/tree/b5f577d6bfdbc543d42638ef8635e3f5ef8cb0c8)，README指向官方仓，许可为 CANN Open Software License 2.0。该提交生产路径 `activation/p_relu_grad_update/op_host/p_relu_grad_update_def.cpp` 注册950/350/960dt，不能直接视为910C实现。experimental/activation/prelu 的 split-C 权重复用线索已落实在 J024/J026；本轮没有复制上游代码。

[Ascend/msopprof 固定版本工具指南](https://github.com/Ascend/msopprof/blob/16beabbbb76f12136cc940866990b7f3e0289602/docs/en/user_guide/msopprof_user_guide.md) 与本机组件对照结果：

| 能力 | 当前可用性与用途 |
|---|---|
| 普通 msprof PipeUtilization | 已用于同机交替 Task/AIV 测量；保留预热和顺序信息 |
| msprof op MemoryDetail | 实板计数器可用；不能提供该架构的核内指令时间线 |
| 实板 Pipe/InstrTimeline | 文档范围不覆盖当前910C/A3，不能假装已采集 |
| simulator | 组件和当前独立 ACLNN 调用入口已可运行；只解释依赖，不当作设备耗时 |
| KernelScale + MetricsProfStart/Stop | 本轮首次实际验证；本机实现显式支持2201，通过编译器 intrinsic 采集局部窗口 |

本机 msopprof 二进制 SHA256 为 `22578c7c9da3bb29bed466ddfc199b33e74e18c12aae3a87ce05c9220a2c8ffe`。API实现中未见显式 PipeBarrier，但不据此假定 instrumentation 完全无扰动。

## J033：补上局部实板证据

J033E 与 J033C 均从 J026 派生，只在每核第二个 batch 设置一个采样窗口。E为空窗口，C包围 Compute；完整算子输入输出和原有同步保留。FP16/FP32，[127,8192,1]，每种两轮反向顺序，101次应用调用、采第101次，msprof op warm-up100、kernel replay；8次运行全部输出正确，40核频率1650MHz。

| dtype / 窗口 | AIV段周期 / μs | Vector段中位数 μs | Scalar段中位数 μs | bank group / bank 冲突比例 |
|---|---:|---:|---:|---:|
| 任一 dtype / 空窗口 | 6 / 0.003636 | 0 | 0.003030 | 0 / 0 |
| FP16 / Compute | 1548 / 0.938182 | 0.781515 | 0.308485 | 21.1240% / 5.9432% |
| FP32 / Compute | 1222 / 0.740606 | 0.540909 | 约0.281 | 26.0229% / 0 |

两轮所有核的段周期与冲突比例一致。结果证明 Compute 窗口中存在可重复的 UB 冲突计数；比例不是互斥的完整算子时间份额。不能将第二批窗口乘批次数、从完整 Task 扣除，或把 E/C 的完整 Task 差别当作优化收益。

## J034：唯一的新优化候选

[官方 A2/A3 UB bank 指南](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0025.html)给出了192KiB UB、48 bank、16 group结构，并以256B源地址错位减少同组读取冲突。结合 J033 的实板计数，选择一次地址布局实验，不做参数扫描。

只修改低精度 split-C 的临时 UB 分配：float scratch 长度向512B对齐，在 mask、fg、gradsFloat、featuresFloat、dxFloat 后各留256B。使同时读取的二元源对在512B周期内相差256B；不假定所有物理 bank 读写冲突都能消除。该规则依赖硬件地址周期和运算依赖，不依赖某个已知 shape。

算式、Cast、Compare/Select、PipeBarrier、队列深度、tile、核数和路由都保持原样。FP32分配走原 constexpr 分支。额外UB最多2816B；低精度最大tile4288时分配最多152896B，小于保留8KiB后的188416B可用空间。J034C为同布局加J033C采样窗口的诊断副本，性能候选为不带采样的J034。

验收顺序：先三dtype741精度和1344逐次输出压力，继而两轮反向分段计数；只有冲突/段时延改善才进行普通 msprof 四轮 ABBA/BAAB，与J026同机比较。完整延迟必须独立通过，不能以分段改善代替。

本文件的最终实测结论和归档校验将在作业完成后附于下节。
