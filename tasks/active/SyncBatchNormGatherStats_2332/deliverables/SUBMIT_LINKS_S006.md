# S006 平台提交清单

云测已过（EXP_S006）：`-e` + `s006_rounding.py` 均 PASS。  
Kernel tip：`b2271b4`。上传以下 7 个源码文件（相对本仓；S006 实质只改了 `.h`）。

- [sync_batch_norm_gather_stats_def.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_def.cpp)
- [sync_batch_norm_gather_stats_infershape.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_infershape.cpp)
- [sync_batch_norm_gather_stats_tiling.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_tiling.cpp)
- [sync_batch_norm_gather_stats.cpp](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats.cpp)
- [sync_batch_norm_gather_stats.h](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats.h)
- [sync_batch_norm_gather_stats_tiling_data.h](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats_tiling_data.h)
- [sync_batch_norm_gather_stats_tiling_key.h](../../../../people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats_tiling_key.h)
