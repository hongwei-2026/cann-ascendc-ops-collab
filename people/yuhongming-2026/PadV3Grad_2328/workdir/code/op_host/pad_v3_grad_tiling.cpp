/*!
 * \file pad_v3_grad_tiling.cpp
 * \brief constant 按输出行分核，直接拷内部；其它模式单核，workspace 只放两块 float 缓冲。
 *        不调用 SyncAll，所以不加 16MB 系统前缀。
 */

#include "pad_v3_grad_parse.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/pad_v3_grad_tiling_data.h"
#include "../op_kernel/pad_v3_grad_tiling_key.h"

namespace optiling {

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

static int64_t Align32(int64_t bytes)
{
    return (bytes + 31) & ~static_cast<int64_t>(31);
}

static ge::graphStatus PadV3GradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    (void)ubSize;

    const gert::StorageShape* xStorage = context->GetInputShape(0);
    const gert::StorageShape* yStorage = context->GetOutputShape(0);
    const gert::Tensor* pads = context->GetInputTensor(1);
    const auto* xDesc = context->GetInputDesc(0);
    const auto* padDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, padDesc);

    const gert::Shape& xShape = xStorage->GetStorageShape();
    const gert::Shape& yShape = yStorage->GetStorageShape();
    int64_t rank = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(rank > kPadV3GradMaxRank, OP_LOGE(context, "rank > 8"), return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* modeStr = attrs->GetAttrPointer<char>(0);
    const bool* contigPtr = attrs->GetAttrPointer<bool>(1);
    const int64_t mode = PadV3GradParseMode(modeStr);
    OP_CHECK_IF(mode < 0, OP_LOGE(context, "unsupported mode"), return ge::GRAPH_FAILED);
    const bool contiguous = contigPtr == nullptr ? true : *contigPtr;

    int64_t front[kPadV3GradMaxRank] = {};
    int64_t back[kPadV3GradMaxRank] = {};
    int64_t xDim[kPadV3GradMaxRank] = {};
    int64_t yDim[kPadV3GradMaxRank] = {};
    if (rank == 0) {
        rank = 1;
        xDim[0] = 1;
        yDim[0] = 1;
    } else {
        const int64_t count = pads->GetShapeSize();
        bool ok = false;
        if (padDesc->GetDataType() == ge::DT_INT64) {
            ok = PadV3GradReadPads(pads->GetData<int64_t>(), count, rank, contiguous, front, back);
        } else {
            ok = PadV3GradReadPads(pads->GetData<int32_t>(), count, rank, contiguous, front, back);
        }
        OP_CHECK_IF(!ok, OP_LOGE(context, "bad paddings"), return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < rank; ++i) {
            xDim[i] = xShape.GetDim(static_cast<size_t>(i));
            OP_CHECK_IF(xDim[i] < 0, OP_LOGE(context, "dynamic dim"), return ge::GRAPH_FAILED);
            yDim[i] = xDim[i] - front[i] - back[i];
            OP_CHECK_IF(yDim[i] < 0, OP_LOGE(context, "pad exceeds dim"), return ge::GRAPH_FAILED);
            const int64_t given = yShape.GetDim(static_cast<size_t>(i));
            OP_CHECK_IF(given >= 0 && given != yDim[i], OP_LOGE(context, "y shape mismatch"), return ge::GRAPH_FAILED);
        }
    }

    int64_t totalX = 1;
    int64_t totalY = 1;
    for (int64_t i = 0; i < rank; ++i) {
        totalX *= xDim[i];
        totalY *= yDim[i];
    }

    const int64_t yRowLength = yDim[rank - 1];
    int64_t usedCores = 1;
    int64_t rowsPerCore = 0;
    int64_t rowRemain = 0;
    if (mode == kPadV3GradModeConstant && yRowLength > 0 && totalY > 0) {
        const int64_t yRows = totalY / yRowLength;
        usedCores = yRows < coreNum ? yRows : coreNum;
        if (usedCores < 1) {
            usedCores = 1;
        }
        rowsPerCore = yRows / usedCores;
        rowRemain = yRows % usedCores;
    }

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    if (mode == kPadV3GradModeConstant || totalX == 0 || totalY == 0) {
        workspace[0] = 0;
    } else {
        workspace[0] = static_cast<size_t>(2 * Align32(totalX * static_cast<int64_t>(sizeof(float))));
    }

    PadV3GradTilingData* tiling = context->GetTilingData<PadV3GradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    *tiling = PadV3GradTilingData{};
    tiling->rank = rank;
    tiling->mode = mode;
    tiling->totalX = totalX;
    tiling->totalY = totalY;
    tiling->usedCores = usedCores;
    tiling->rowsPerCore = rowsPerCore;
    tiling->rowRemain = rowRemain;
    tiling->yRowLength = yRowLength;
    for (int64_t i = 0; i < kPadV3GradMaxRank; ++i) {
        tiling->xDim[i] = xDim[i];
        tiling->yDim[i] = yDim[i];
        tiling->front[i] = front[i];
        tiling->back[i] = back[i];
    }

    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    uint64_t tilingKey = GET_TPL_TILING_KEY(PADV3GRAD_TPL_SCH_MODE_1);
    const auto dtype = xDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(PADV3GRAD_TPL_SCH_MODE_0);
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(PADV3GRAD_TPL_SCH_MODE_2);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPadV3Grad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct PadV3GradCompileInfo {};

IMPL_OP_OPTILING(PadV3Grad).Tiling(PadV3GradTilingFunc).TilingParse<PadV3GradCompileInfo>(TilingParseForPadV3Grad);

} // namespace optiling
