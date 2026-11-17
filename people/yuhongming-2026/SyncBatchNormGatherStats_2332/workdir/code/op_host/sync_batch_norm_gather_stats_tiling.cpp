/*!
 * \file sync_batch_norm_gather_stats_tiling.cpp
 * \brief SyncBatchNormGatherStats 算子 Tiling 实现
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sync_batch_norm_gather_stats_tiling_data.h"
#include "../op_kernel/sync_batch_norm_gather_stats_tiling_key.h"

namespace optiling {

namespace {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t kBufCount = 7;          // raw/load/accSum/accSq/mean/var/tmp
constexpr int64_t kReserveBytes = 4 * 1024;

int64_t CeilDivI64(int64_t value, int64_t divisor)
{
    if (divisor <= 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

int64_t FloatToBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<int64_t>(bits);
}

} // namespace

static ge::graphStatus SyncBatchNormGatherStatsTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* sumStorage = context->GetInputShape(0);
    const gert::StorageShape* countStorage = context->GetInputShape(2);
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* countDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, countStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, countDesc);

    const gert::Shape& sumShape = sumStorage->GetStorageShape();
    const int64_t rank = static_cast<int64_t>(sumShape.GetDimNum());
    int64_t channels = 1;
    int64_t worldSize = 1;
    if (rank <= 0) {
        channels = 1;
        worldSize = 1;
    } else if (rank == 1) {
        channels = sumShape.GetDim(0);
        worldSize = 1;
    } else {
        channels = sumShape.GetDim(rank - 1);
        worldSize = 1;
        for (int64_t i = 0; i < rank - 1; ++i) {
            worldSize *= sumShape.GetDim(i);
        }
    }
    OP_CHECK_IF(channels < 0 || worldSize <= 0, OP_LOGE(context, "invalid sum shape"), return ge::GRAPH_FAILED);

    int64_t countNum = countStorage->GetStorageShape().GetShapeSize();
    if (countNum < 0) {
        countNum = 0;
    }
    const int64_t countIsInt64 = (countDesc->GetDataType() == ge::DT_INT64) ? 1 : 0;

    float momentum = 0.1f;
    float eps = 1e-5f;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* momentumPtr = attrs->GetAttrPointer<float>(0);
        const float* epsPtr = attrs->GetAttrPointer<float>(1);
        if (momentumPtr != nullptr) {
            momentum = *momentumPtr;
        }
        if (epsPtr != nullptr) {
            eps = *epsPtr;
        }
    }

    int64_t budget = static_cast<int64_t>(ubSize) - kReserveBytes;
    if (budget < 4096) {
        budget = static_cast<int64_t>(ubSize) / 2;
    }
    int64_t maxTile = budget / (kBufCount * static_cast<int64_t>(sizeof(float)));
    maxTile = (maxTile / 8) * 8; // fp32 32B 对齐
    if (maxTile < 64) {
        maxTile = 64;
    }
    if (maxTile > 8192) {
        maxTile = 8192;
    }

    // 小 C 场景少核粗粒度更好（单通道分核会触发极短向量且开销更大）。
    // 每核至少处理 8 个通道（fp32 32B），不足则合并到更少核。
    constexpr int64_t kMinBlock = 8;
    int64_t usedCores = coreNum;
    if (channels < usedCores) {
        usedCores = channels > 0 ? channels : 1;
    }
    if (usedCores < 1) {
        usedCores = 1;
    }
    int64_t blockFactor = channels > 0 ? CeilDivI64(channels, usedCores) : 1;
    if (blockFactor < kMinBlock && channels > 0) {
        blockFactor = channels < kMinBlock ? channels : kMinBlock;
        usedCores = CeilDivI64(channels, blockFactor);
    }
    if (blockFactor < 1) {
        blockFactor = 1;
    }
    if (channels > 0) {
        usedCores = CeilDivI64(channels, blockFactor);
    }
    int64_t ubFactor = blockFactor < maxTile ? blockFactor : maxTile;
    if (ubFactor < 1) {
        ubFactor = 1;
    }
    // ubFactor 向上对齐到 8，便于向量指令
    ubFactor = ((ubFactor + 7) / 8) * 8;
    if (ubFactor > maxTile) {
        ubFactor = maxTile;
    }

    SyncBatchNormGatherStatsTilingData* tiling = context->GetTilingData<SyncBatchNormGatherStatsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = channels;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->worldSize = worldSize;
    tiling->countNum = countNum;
    tiling->countIsInt64 = countIsInt64;
    tiling->momentumBits = FloatToBits(momentum);
    tiling->epsBits = FloatToBits(eps);

    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    uint64_t tilingKey = 0;
    const ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(SYNCBATCHNORMGATHERSTATS_TPL_SCH_MODE_0);
    } else if (dtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(SYNCBATCHNORMGATHERSTATS_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "unsupported dtype %d", static_cast<int>(dtype));
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSyncBatchNormGatherStats([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SyncBatchNormGatherStatsCompileInfo {};

IMPL_OP_OPTILING(SyncBatchNormGatherStats)
    .Tiling(SyncBatchNormGatherStatsTilingFunc)
    .TilingParse<SyncBatchNormGatherStatsCompileInfo>(TilingParseForSyncBatchNormGatherStats);

} // namespace optiling
