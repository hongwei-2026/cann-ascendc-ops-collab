# J046：同切分、同遍历的权重常驻 / 逐 tile 重读对照

基线 J035，唯一文件 `op_kernel/p_relu_grad_update_splitc.h`，仅两处源码改动：`residentWeights_=true` 改为 false，删除 Process 结束时驻留权重的最终 FreeTensor。Host、Ctile、核数、batch 分组和遍历、UB 分配顺序及大小、数学计算、Cast 和显式 PipeBarrier 代码均保持原样。逐 tile 改走原有 nonresident 的 DataCopyPad/EnQue/DeQue/FreeTensor 路径，队列同步生命周期随此策略改变。

因此本实验隔离的是**权重搬运与队列生命周期策略**，不单独隔离 HBM 字节；编译器生成的该分支指令也会变化。与 J045 不同，J046 没有改变 flat/C/batch 遍历顺序。J035、J038、J045 均保留，未操作平台或推送代码。

## 环境及验证口径

实板 Ascend910_9362，编译目标 ascend910_93，CANN 9.0.0，驱动 25.5.5，实际 Host 使用最多 40 AIV。启动前已核上一轮设备作业退出 0，npu-smi 无运行 NPU 进程。第一次只读 npu-smi 因未加载库路径失败，加载既有 CANN/driver 会话环境后正常；未安装/修改系统配置。

源码比较证明其他 45 个文件逐字不变。ELF 证明每 dtype 15 个入口仅 split-C 对应函数不同，其余 14 个函数字节相同。split-C 函数大小 FP32 6608→6552 bytes、FP16 7312→7240、BF16 7036→6964；代码变小本身不证明运行更快。

设备精度：780/780 有限数值案例通过，覆盖 FP32/FP16/BF16、标量/每通道权重、不同 N/C/S、尾块和多种符号/零模式。压力：57 案例、5,184 次调用，每次预填输出后检查两个输出，共 10,368 次输出检查通过。未宣称所有 IEEE 特殊值覆盖。

普通性能：36 输入（三 dtype，12 小输入、18 原大输入及 6 个通道尾块输入），4 轮 ABBA/BAAB/ABBA/BAAB；后两轮反转输入顺序。每组 250 次，取第 101–250 次 Task Duration 中位数，平均同轮两个位置，再平均四轮。576 组、144,000 条 kernel 记录，含未修改路径控制。组内输出检查与输入 SHA 校验保留。

无 profiler：12 输入、四轮交替，每窗口预热 4×128 次，再记录 12 个连续 enqueue 批次，后 10 个 event 批间隔/128 的中位数。192 窗口、2,304 个 event 样本；该值是连续提交间隔，不是隔离 kernel 的执行时长，不与普通 profiler 混算。

实板 MemoryDetail：本机 `msprof op --help` 已核参数。目标三 dtype `[127,8192,1]` 加 FP16 尾块 `[81,8256,1]`，按 J035/J046/J046/J035 四位置采集，共 16 次。固定 application replay，skip=100、launch-count=1、warm-up=100。它会多次 replay 收集计数器；计数器用于解释搬运行为，采集中的 Task 不代替普通四轮性能结论。没有使用 simulator/graph PMU 数据。

## 权重搬运：逻辑次数与实测计数器分开

`[127,8192,1]` 三 dtype：Ctile=4096、两块、20 个 batch 组、40 核；26 核处理 6 tile、14 核处理 7 tile。每 tile 的 C 区间相同，无 C 尾块。J035 每核只读取一次权重，J046 每 tile 读取，共 **40→254 次权重 DataCopyPad**。权重队列 Alloc/EnQue/DeQue/Free 各自也为 40→254 次。grads/features 读取总计 508 次、两个输出写回总计 508 次，均不变。

FP32 权重逻辑字节 655,360→4,161,536；FP16/BF16 327,680→2,080,768。全部逻辑 GM 读写约增加 20.27%。两版本低精度权重 Cast 仍是每 tile 一次，共 254 次；没有同时复用转换结果。

这些由源代码和每核任务映射推导，不是实测 HBM 事务。完整 111 个适用精度案例的每核任务与逻辑计数在 `J046-logical-counts.json`。PMU 的 MTE2 指令数并不与源代码 DataCopy 调用一一对应，包含生成指令/其他读取；GM_to_UB、main_memory 字段按工具原名保留，不重命名为纯权重或纯 HBM 流量。[官方字段说明](https://github.com/Ascend/msopprof/blob/16beabbbb76f12136cc940866990b7f3e0289602/docs/en/user_guide/msopprof_performance_data.md) 将 MTE2 字段定义为指令条数，将 GM_to_UB 定义为 GM 到 UB 搬运量；这是计数器含义依据，不是本轮结果来源。

L2 命中率只描述本轮 application replay 和相应输入的采集状态，不据此推断 L2 容量、访问延迟、冷启动性能或比赛条件。MTE2/Vector 等活跃时间允许重叠，不能相加当总时延，也不是独占耗时占比。

官方基线：用户确认 1148578 源码与 J035 一致，五点 Pass、7.15/7.29/7.11/5.88/6.06 μs；确认来源为用户，本轮未下载官方提交快照。五点 shape/硬件/统计口径仍未知，以下自选输入不得映射为官方某点。平台由用户操作。

