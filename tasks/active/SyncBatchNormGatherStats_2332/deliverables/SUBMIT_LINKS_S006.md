# S006 平台提交清单

先按 `../CLOUD-S006.md` 跑 910B 样例与舍入边界测试。完成后上传以下 7 个源码文件；S006 只改了 Kernel 的头文件。

- [sync_batch_norm_gather_stats_def.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_def.cpp)
- [sync_batch_norm_gather_stats_infershape.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_infershape.cpp)
- [sync_batch_norm_gather_stats_tiling.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_tiling.cpp)
- [sync_batch_norm_gather_stats.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats.cpp)
- [sync_batch_norm_gather_stats.h](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats.h)
- [sync_batch_norm_gather_stats_tiling_data.h](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats_tiling_data.h)
- [sync_batch_norm_gather_stats_tiling_key.h](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats_tiling_key.h)
