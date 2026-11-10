# 下一项严格权重生命周期对照（设计，尚未实现/运行）

目的：区分 J035 split-C 的跨 batch UB 权重常驻是否有益。J045 改了遍历顺序，不能单独回答此问题。

基线 J035，不更改 Host、tile、核数、UB 布局、batchStart/batchEnd、channelOffset、Compute 算术。只设 residentWeights_=false，保留原 Process 的 batch 遍历；每次 CopyIn 将走原 nonresident 权重 DataCopy/EnQue，Compute 每次 DeQue 并 FreeTensor；删除 Process 末尾针对 residentWeightsLocal_ 的最终释放。这改变权重读入与队列生命周期，不能声称只增加 HBM 流量。firstTile_ 不再影响非驻留权重分支，避免另做无关清理。

保留低精度每 tile Cast 行为，暂不同时复用转换结果。先在原 780 精度、57 压力案例确认输出，再做 J035/新候选四轮交替、同输入、相同块分布，包含小输入未改路径控制。有限重点输入：[127,8192,1]、[91,10368,1] 和通道尾块/三 dtype；仅在差异大于未改路径漂移且跨轮同向时解释收益。无收益即结束本假设。

L2 命中、容量和延迟未测，不以“权重小、必在 L2”替代测量，也不把逻辑 GM 字节当实测 DRAM 流量。不提交平台、不改在线草稿；候选精度验证后可提供精确文件供用户决定。

这是后续方案，不代表已建立 J046 或已启动设备作业。
