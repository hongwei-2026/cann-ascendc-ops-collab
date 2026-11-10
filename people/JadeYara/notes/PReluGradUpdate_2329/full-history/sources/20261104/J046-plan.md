# J046：同切分/同遍历的权重生命周期对照

基于冻结 J035，仅将 split-C residentWeights_=true 改 false，并删除 Process 末尾驻留 tensor 的释放。保留 Host、Ctile、batchStart/End、batch 遍历、blockDim、UB 分配布局、算术、低精度每 tile Cast、其他路径。使用原 nonresident 的每 tile 权重 DataCopy/EnQue/DeQue/FreeTensor。改变的是权重搬运和队列生命周期整体，不能把效果仅归因于 HBM 字节。

先 780 三 dtype 精度 + 57 案例 5184 次双输出压力，再沿用 J045 的 30 个小/大输入并加入两种 C 尾块形状三 dtype，共 36 个输入，四轮 ABBA/BAAB，后两轮反转输入顺序。普通 PipeUtilization 记录 Task、AIV/Scalar/Vector/MTE 活跃时间、cycle、Block Num；未改路径作漂移控制。有限无 profiler enqueue 对照用于交叉核查，口径独立。统计权重搬运/队列次数为基于执行路径的逻辑计数，不冒充 PMU 实际 HBM 事务；不预设 L2。

用户已确认官方 1148578=J035，官方输入未知，用户负责提交；本轮不操作平台、不提交、不推送。所有原版本和证据保留。
