/*!
 * \file sync_batch_norm_gather_stats_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SYNCBATCHNORMGATHERSTATS_TILING_DATA_H_
#define _SYNCBATCHNORMGATHERSTATS_TILING_DATA_H_

struct SyncBatchNormGatherStatsTilingData {
    int64_t totalNum = 0;       // 通道数 C
    int64_t blockFactor = 1;    // 每个核处理的通道数
    int64_t ubFactor = 0;       // 每次 UB 循环处理的通道数
    int64_t worldSize = 1;      // 设备维 N，total_sum 按 [N, C] 排布
    int64_t countNum = 1;       // sample_count 元素个数
    int64_t countIsInt64 = 0;   // 0: int32, 1: int64
    int64_t momentumBits = 0;   // momentum 的 float32 位型
    int64_t epsBits = 0;        // eps 的 float32 位型
};
#endif
