# 平台粘贴用本地超链接（S003 · tip `18e8ca5`）

相对 S002：仅 `sync_batch_norm_gather_stats.h` 在 fp16 写出前重写 NaN/±inf。  
**须先云 `-e` 回归，再上传。** 云/平台未绿前勿当正式 Pass。

## Host（未改）

- [sync_batch_norm_gather_stats_def.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_def.cpp)
- [sync_batch_norm_gather_stats_infershape.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_infershape.cpp)
- [sync_batch_norm_gather_stats_tiling.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_tiling.cpp)

## Kernel

- [sync_batch_norm_gather_stats.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats.cpp)
- [sync_batch_norm_gather_stats.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats.h) ← **S003 改动**
- [sync_batch_norm_gather_stats_tiling_data.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats_tiling_data.h)
- [sync_batch_norm_gather_stats_tiling_key.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats_tiling_key.h)
