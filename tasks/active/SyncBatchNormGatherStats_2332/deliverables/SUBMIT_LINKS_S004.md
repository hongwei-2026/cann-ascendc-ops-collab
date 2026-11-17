# 平台粘贴用本地超链接（S004 · tip `fa21ed5`）

云 EXP_S004 **`-e` PASS**（0/6）。相对 S003：`QuantFp16Vec` 保 NaN/±Inf
（`V_S`+`S_V`）；`StoreAsInputDtype` 保留 S003 并补 `S_MTE3`。
**平台 50 点结果待本清单重交后确认。**

## Host（未修改）

- [sync_batch_norm_gather_stats_def.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_def.cpp)
- [sync_batch_norm_gather_stats_infershape.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_infershape.cpp)
- [sync_batch_norm_gather_stats_tiling.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_host/sync_batch_norm_gather_stats_tiling.cpp)

## Kernel

- [sync_batch_norm_gather_stats.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats.cpp)
- [sync_batch_norm_gather_stats.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats.h) → **S004 修改**
- [sync_batch_norm_gather_stats_tiling_data.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats_tiling_data.h)
- [sync_batch_norm_gather_stats_tiling_key.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/op_kernel/sync_batch_norm_gather_stats_tiling_key.h)

先在 910B 环境完成 `build.sh -j8 -e`，再上传以上 7 个文件；本清单不代表平台已通过。
