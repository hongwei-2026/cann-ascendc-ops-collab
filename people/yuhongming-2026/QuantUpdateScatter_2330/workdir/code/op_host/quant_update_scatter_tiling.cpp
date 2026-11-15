/*!
 * \file quant_update_scatter_tiling.cpp
 * \brief QuantUpdateScatter 算子 Tiling 实现
 */

#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/quant_update_scatter_tiling_data.h"
#include "../op_kernel/quant_update_scatter_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

namespace {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t kReserveBytes = 8 * 1024;
constexpr int64_t kIndexCacheBytes = 2048 * sizeof(int64_t);
constexpr int64_t kCopyBytesPerCore = 32 * 1024;
constexpr int64_t kDmaBlockBytes = 32;
// Worst case: two updates/output slots plus all parameter/calculation buffers.
// Single-row direct buffers use less memory but share this safe upper bound.
constexpr int64_t kBufBytesPerElem = 2 * 2 + 4 + 4 + 4 + 4 + 4 + 2 + 2 + 2 * 1;

int64_t ShapeSize(const gert::Shape& shape)
{
    int64_t n = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        n *= shape.GetDim(i);
    }
    return n;
}

int64_t NormalizeAxis(int64_t axis, int64_t rank)
{
    if (rank <= 0) {
        return 0;
    }
    int64_t ax = axis;
    if (ax < 0) {
        ax += rank;
    }
    if (ax < 0) {
        ax = 0;
    }
    if (ax >= rank) {
        ax = rank - 1;
    }
    return ax;
}

} // namespace

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus QuantUpdateScatterTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* varStorage = context->GetInputShape(0);
    const gert::StorageShape* indicesStorage = context->GetInputShape(1);
    const gert::StorageShape* updatesStorage = context->GetInputShape(2);
    const gert::StorageShape* scalesStorage = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, varStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalesStorage);

    const auto* updatesDesc = context->GetInputDesc(2);
    const auto* indicesDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);

    const gert::Shape& varShape = varStorage->GetStorageShape();
    const gert::Shape& updatesShape = updatesStorage->GetStorageShape();
    const gert::Shape& indicesShape = indicesStorage->GetStorageShape();

    const int64_t updatesRank = static_cast<int64_t>(updatesShape.GetDimNum());
    const int64_t varRank = static_cast<int64_t>(varShape.GetDimNum());
    OP_CHECK_IF(updatesRank < 1 || varRank < 1, OP_LOGE(context, "rank < 1"), return ge::GRAPH_FAILED);

    int64_t axis = -2;
    int64_t quantAxis = -1;
    bool reciprocal = false;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        // Attr order: reduce(0), axis(1), quant_axis(2), reciprocal_scale(3)
        const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(1);
        const int64_t* quantAxisPtr = attrs->GetAttrPointer<int64_t>(2);
        const bool* recipPtr = attrs->GetAttrPointer<bool>(3);
        if (axisPtr != nullptr) {
            axis = *axisPtr;
        }
        if (quantAxisPtr != nullptr) {
            quantAxis = *quantAxisPtr;
        }
        if (recipPtr != nullptr) {
            reciprocal = *recipPtr;
        }
    }
    axis = NormalizeAxis(axis, updatesRank);
    quantAxis = NormalizeAxis(quantAxis, updatesRank);
    OP_CHECK_IF(
        quantAxis != updatesRank - 1, OP_LOGE(context, "only quant_axis=-1 supported"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(axis != updatesRank - 2 && updatesRank >= 2, OP_LOGE(context, "only axis=-2 supported"),
        return ge::GRAPH_FAILED);
    // rank==1: treat whole vector as lastDim, axis size 1
    const bool rank1 = (updatesRank == 1);

    const int64_t lastDim = updatesShape.GetDim(updatesRank - 1);
    const int64_t updateAxis = rank1 ? 1 : updatesShape.GetDim(axis);
    const int64_t varAxis = rank1 ? 1 : (varRank >= 2 ? varShape.GetDim(varRank - 2) : 1);
    const int64_t batchSize = updatesShape.GetDim(0);
    int64_t midSize = 1;
    if (!rank1) {
        for (int64_t i = 1; i < axis; ++i) {
            midSize *= updatesShape.GetDim(i);
        }
    }
    if (midSize <= 0) {
        midSize = 1;
    }

    const int64_t totalNum = ShapeSize(varShape);
    const int64_t indexElements = ShapeSize(indicesShape);
    bool hasZeroPoints = (context->GetOptionalInputShape(4) != nullptr);
    if (!hasZeroPoints && context->GetInputShape(4) != nullptr) {
        hasZeroPoints = true;
    }
    const ge::DataType indexType = indicesDesc->GetDataType();
    OP_CHECK_IF(indexType != ge::DT_INT32 && indexType != ge::DT_INT64,
        OP_LOGE(context, "indices must be int32 or int64"), return ge::GRAPH_FAILED);
    const int64_t indicesIsInt64 = indexType == ge::DT_INT64 ? 1 : 0;

    const int64_t budget = static_cast<int64_t>(ubSize) - kReserveBytes - kIndexCacheBytes;
    OP_CHECK_IF(budget < kDmaBlockBytes * kBufBytesPerElem,
        OP_LOGE(context, "UB too small for aligned buffers and indices cache"), return ge::GRAPH_FAILED);
    int64_t maxTile = (budget / kBufBytesPerElem / kDmaBlockBytes) * kDmaBlockBytes;
    maxTile = std::min(maxTile, int64_t{8192});
    int64_t ubFactor = std::min(lastDim > 0 ? lastDim : int64_t{1}, maxTile);
    ubFactor = CeilDiv(ubFactor, kDmaBlockBytes) * kDmaBlockBytes;
    const int64_t batchCores = std::min(coreNum, batchSize > 0 ? batchSize : int64_t{1});
    const int64_t copyCores = std::min(coreNum,
        totalNum > 0 ? CeilDiv(totalNum, kCopyBytesPerCore) : int64_t{1});
    // Do not recompute usedCores from batchFactor: copy-only cores are intentional.
    const int64_t usedCores = std::max(batchCores, copyCores);
    const int64_t blockFactor = batchSize > 0 ? CeilDiv(batchSize, usedCores) : 1;

    QuantUpdateScatterTilingData* tiling = context->GetTilingData<QuantUpdateScatterTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->coreNum = usedCores;
    tiling->batchSize = batchSize;
    tiling->midSize = midSize;
    tiling->updateAxis = updateAxis;
    tiling->varAxis = varAxis;
    tiling->lastDim = lastDim;
    tiling->indexElements = indexElements;
    tiling->hasZeroPoints = hasZeroPoints ? 1 : 0;
    tiling->reciprocalScale = reciprocal ? 1 : 0;
    tiling->indicesIsInt64 = indicesIsInt64;

    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    // Modes describe the quantization triple; index width is independent.
    uint64_t tilingKey = 0;
    const ge::DataType updatesDtype = updatesDesc->GetDataType();
    if (updatesDtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(QUANTUPDATESCATTER_TPL_SCH_MODE_0);
    } else if (updatesDtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(QUANTUPDATESCATTER_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "unsupported updates dtype %d", static_cast<int>(updatesDtype));
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);
    (void)scalesStorage;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantUpdateScatter([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantUpdateScatterCompileInfo {};

IMPL_OP_OPTILING(QuantUpdateScatter)
    .Tiling(QuantUpdateScatterTilingFunc)
    .TilingParse<QuantUpdateScatterCompileInfo>(TilingParseForQuantUpdateScatter);

} // namespace optiling
