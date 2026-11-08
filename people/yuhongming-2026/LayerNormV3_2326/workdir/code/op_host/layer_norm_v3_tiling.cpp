/*!
 * \file layer_norm_v3_tiling.cpp
 * \brief 按归一化行分核。UB 放 6 块 float 缓冲（x/tmp/gamma/beta/work/raw）。
 */

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/layer_norm_v3_tiling_data.h"
#include "../op_kernel/layer_norm_v3_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr int64_t kBufCount = 6;
constexpr int64_t kAlign = 64;
constexpr int64_t kTileCap = 1024;
constexpr int64_t kUbReserve = 32 * 1024;

static int64_t NormalizeAxis(int64_t axis, int64_t rank)
{
    if (axis < 0) {
        axis += rank;
    }
    return axis;
}

static int64_t ShapeProduct(const gert::Shape& shape, int64_t begin, int64_t end)
{
    if (begin >= end) {
        return 1;
    }
    int64_t prod = 1;
    for (int64_t i = begin; i < end; ++i) {
        const int64_t dim = shape.GetDim(static_cast<size_t>(i));
        if (dim < 0) {
            return -1;
        }
        prod *= dim;
    }
    return prod;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = ascendcPlatform.GetCoreNum();
    }
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LayerNormV3TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = WS_SYS_SIZE;

    const gert::StorageShape* xStorage = context->GetInputShape(0);
    const gert::StorageShape* gammaStorage = context->GetInputShape(1);
    const gert::StorageShape* betaStorage = context->GetInputShape(2);
    const auto* xDesc = context->GetInputDesc(0);
    const auto* gammaDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDesc);

    const gert::Shape& xShape = xStorage->GetStorageShape();
    const gert::Shape& gammaShape = gammaStorage->GetStorageShape();
    const gert::Shape& betaShape = betaStorage->GetStorageShape();
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());

    int64_t beginNorm = 0;
    int64_t beginParams = 0;
    float epsilon = 1e-5f;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* normPtr = attrs->GetAttrPointer<int64_t>(0);
        const int64_t* paramsPtr = attrs->GetAttrPointer<int64_t>(1);
        const float* epsPtr = attrs->GetAttrPointer<float>(2);
        if (normPtr != nullptr) {
            beginNorm = *normPtr;
        }
        if (paramsPtr != nullptr) {
            beginParams = *paramsPtr;
        }
        if (epsPtr != nullptr) {
            epsilon = *epsPtr;
        }
    }
    if (rank > 0) {
        beginNorm = NormalizeAxis(beginNorm, rank);
        beginParams = NormalizeAxis(beginParams, rank);
        OP_CHECK_IF(beginNorm < 0 || beginNorm > rank, OP_LOGE(context, "begin_norm_axis out of range"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(beginParams < 0 || beginParams > rank, OP_LOGE(context, "begin_params_axis out of range"),
            return ge::GRAPH_FAILED);
    } else {
        beginNorm = 0;
        beginParams = 0;
    }

    const int64_t rowNum = ShapeProduct(xShape, 0, beginNorm);
    const int64_t colNum = ShapeProduct(xShape, beginNorm, rank);
    const int64_t gammaNum = ShapeProduct(gammaShape, 0, static_cast<int64_t>(gammaShape.GetDimNum()));
    const int64_t betaNum = ShapeProduct(betaShape, 0, static_cast<int64_t>(betaShape.GetDimNum()));
    const int64_t expectGamma = ShapeProduct(xShape, beginParams, rank);
    OP_CHECK_IF(rowNum <= 0 || colNum <= 0 || gammaNum <= 0 || betaNum <= 0,
        OP_LOGE(context, "invalid shape product"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gammaNum != betaNum, OP_LOGE(context, "gamma and beta numel mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(expectGamma > 0 && gammaNum != expectGamma, OP_LOGE(context, "gamma numel != x[begin_params_axis:]"),
        return ge::GRAPH_FAILED);

    int64_t budget = static_cast<int64_t>(ubSize) - kUbReserve;
    if (budget < 4096) {
        budget = static_cast<int64_t>(ubSize) / 2;
    }
    int64_t maxAlign = budget / (kBufCount * static_cast<int64_t>(sizeof(float)));
    maxAlign = maxAlign / kAlign * kAlign;
    if (maxAlign < kAlign) {
        maxAlign = kAlign;
    }
    if (maxAlign > kTileCap) {
        maxAlign = kTileCap;
    }
    int64_t tileAlign = maxAlign;
    if (colNum < tileAlign) {
        tileAlign = (colNum + kAlign - 1) / kAlign * kAlign;
    }
    if (tileAlign < kAlign) {
        tileAlign = kAlign;
    }
    const int64_t tileLength = colNum < tileAlign ? colNum : tileAlign;

    int64_t usedCores = std::min(coreNum, rowNum);
    if (usedCores <= 0) {
        usedCores = 1;
    }
    const int64_t rowsPerCore = rowNum / usedCores;
    const int64_t rowRemain = rowNum % usedCores;

    LayerNormV3TilingData* tiling = context->GetTilingData<LayerNormV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->rowNum = rowNum;
    tiling->colNum = colNum;
    tiling->gammaNum = gammaNum;
    tiling->rowsPerCore = rowsPerCore;
    tiling->rowRemain = rowRemain;
    tiling->usedCores = usedCores;
    tiling->tileLength = tileLength;
    tiling->tileAlign = tileAlign;
    uint32_t epsBits = 0;
    std::memcpy(&epsBits, &epsilon, sizeof(float));
    tiling->epsilonBits = static_cast<int64_t>(epsBits);

    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    const auto xDtype = xDesc->GetDataType();
    const auto gammaDtype = gammaDesc->GetDataType();
    uint64_t tilingKey = GET_TPL_TILING_KEY(LAYERNORMV3_TPL_SCH_MODE_1);
    if (xDtype == ge::DT_FLOAT16 && gammaDtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(LAYERNORMV3_TPL_SCH_MODE_3);
    } else if (xDtype == ge::DT_BF16 && gammaDtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(LAYERNORMV3_TPL_SCH_MODE_4);
    } else if (xDtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(LAYERNORMV3_TPL_SCH_MODE_0);
    } else if (xDtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(LAYERNORMV3_TPL_SCH_MODE_2);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLayerNormV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LayerNormV3CompileInfo {};

IMPL_OP_OPTILING(LayerNormV3).Tiling(LayerNormV3TilingFunc).TilingParse<LayerNormV3CompileInfo>(TilingParseForLayerNormV3);

} // namespace optiling
