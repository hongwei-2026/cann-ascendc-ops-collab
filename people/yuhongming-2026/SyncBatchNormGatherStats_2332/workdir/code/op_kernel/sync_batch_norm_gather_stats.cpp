/*!
 * \file sync_batch_norm_gather_stats.cpp
 * \brief SyncBatchNormGatherStats 算子 kernel 入口
 */

#include "sync_batch_norm_gather_stats.h"

enum class SyncBatchNormGatherStatsTilingKey : uint32_t
{
    TILING_KEY_SYNCBATCHNORMGATHERSTATS_MODE_0 = 0,
    TILING_KEY_SYNCBATCHNORMGATHERSTATS_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void sync_batch_norm_gather_stats(GM_ADDR total_sum, GM_ADDR total_square_sum, GM_ADDR sample_count, GM_ADDR mean, GM_ADDR variance, GM_ADDR batch_mean, GM_ADDR batch_invstd, GM_ADDR new_mean, GM_ADDR new_variance, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SyncBatchNormGatherStatsTilingData);
    GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(SyncBatchNormGatherStatsTilingKey::TILING_KEY_SYNCBATCHNORMGATHERSTATS_MODE_0)) {
        NsSyncBatchNormGatherStats::SyncBatchNormGatherStats<half> op;
        op.Init(total_sum, total_square_sum, sample_count, mean, variance, batch_mean, batch_invstd, new_mean, new_variance, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SyncBatchNormGatherStatsTilingKey::TILING_KEY_SYNCBATCHNORMGATHERSTATS_MODE_1)) {
        NsSyncBatchNormGatherStats::SyncBatchNormGatherStats<float> op;
        op.Init(total_sum, total_square_sum, sample_count, mean, variance, batch_mean, batch_invstd, new_mean, new_variance, &tilingData);
        op.Process();
    }
}
