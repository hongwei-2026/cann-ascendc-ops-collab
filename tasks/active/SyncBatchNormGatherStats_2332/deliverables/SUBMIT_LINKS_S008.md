# S008 平台提交清单

910B 构建和首样例通过；大 C 的 batch_mean 写零已在三次运行中消失。
严格矩阵仍有 invstd 7 处和 new_variance 13 处超差；按样例比例规则外推
均低于阈值，但平台规则未知。平台结果不能预先判定。

上传以下 7 个源码文件，使用含 S008 同步的版本（Kernel commit `140dbdf`）：

- [sync_batch_norm_gather_stats_def.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_def.cpp)
- [sync_batch_norm_gather_stats_infershape.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_infershape.cpp)
- [sync_batch_norm_gather_stats_tiling.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_tiling.cpp)
- [sync_batch_norm_gather_stats.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats.cpp)
- [sync_batch_norm_gather_stats.h](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats.h)
- [sync_batch_norm_gather_stats_tiling_data.h](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats_tiling_data.h)
- [sync_batch_norm_gather_stats_tiling_key.h](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats_tiling_key.h)
