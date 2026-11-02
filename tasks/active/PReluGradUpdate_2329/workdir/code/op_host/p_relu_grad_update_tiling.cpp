/*!
 * \file p_relu_grad_update_tiling.cpp
 * \brief Tiling — design_final §3.3
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/p_relu_grad_update_tiling_data.h"
#include "../op_kernel/p_relu_grad_update_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

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

static int64_t DtypeSize(ge::DataType dt)
{
    if (dt == ge::DT_FLOAT) {
        return 4;
    }
    return 2;
}

static int64_t UbCoef(int64_t typeSize)
{
    return (typeSize == 4) ? 42 : 74;
}

static ge::graphStatus PReluGradUpdateTilingFunc(gert::TilingContext* context)
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

    PReluGradUpdateTilingData* tiling = context->GetTilingData<PReluGradUpdateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto shapePtr = context->GetInputShape(0);
    auto wShapePtr = context->GetInputShape(2);
    auto desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);

    const gert::Shape& shape = EnsureNotScalar(shapePtr->GetStorageShape());
    const gert::Shape& wShape = EnsureNotScalar(wShapePtr->GetStorageShape());
    OP_CHECK_IF(shape.GetDimNum() < 2, OP_LOGE(context, "rank < 2"), return ge::GRAPH_FAILED);

    const int64_t n = shape.GetDim(0);
    const int64_t c = shape.GetDim(1);
    int64_t inner = 1;
    for (size_t i = 2; i < shape.GetDimNum(); ++i) {
        inner *= shape.GetDim(i);
    }
    const int64_t M = n * c * inner;
    const int64_t S = c * inner;
    const int64_t wLen = wShape.GetDim(0);
    const int64_t weightsPerChannel = (wLen == 1) ? 0 : 1;

    const int64_t typeSize = DtypeSize(desc->GetDataType());
    const int64_t alignElem = (typeSize > 0) ? (32 / typeSize) : 8;
    const int64_t coef = UbCoef(typeSize);
    int64_t maxTile = FloorAlign(static_cast<int64_t>((ubSize * 85) / 100) / coef, alignElem);
    if (maxTile < alignElem) {
        maxTile = alignElem;
    }

    int64_t tileLen = 0;
    int64_t rowsPerTile = 0;
    if (S > 0 && S <= maxTile) {
        rowsPerTile = n;
        if (S > 0 && rowsPerTile > maxTile / S) {
            rowsPerTile = maxTile / S;
        }
        if (rowsPerTile < 1) {
            rowsPerTile = 1;
        }
        tileLen = rowsPerTile * S;
    } else {
        rowsPerTile = 0;
        tileLen = maxTile;
    }

    int64_t perCore = CeilAlign(CeilDiv((M > 0) ? M : static_cast<int64_t>(1), coreNum), alignElem);
    if (rowsPerTile > 0 && S * typeSize >= 32) {
        perCore = CeilAlign(perCore, S);
    }
    int64_t usedCores = CeilDiv((M > 0) ? M : static_cast<int64_t>(1), perCore);
    if (usedCores > coreNum) {
        usedCores = coreNum;
    }
    if (usedCores < 1) {
        usedCores = 1;
    }
    if (M == 0) {
        usedCores = 1;
        perCore = 0;
        tileLen = alignElem;
    }

    tiling->totalNum = M;
    tiling->coreNum = usedCores;
    tiling->blockFactor = perCore;
    tiling->tileLen = tileLen;
    tiling->channelNum = c;
    tiling->innerSize = inner;
    tiling->sampleSize = S;
    tiling->weightsPerChannel = weightsPerChannel;
    tiling->rowsPerTile = rowsPerTile;

    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    uint64_t tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_SCH_MODE_0);
    if (desc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_SCH_MODE_1);
    } else if (desc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_SCH_MODE_2);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPReluGradUpdate([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct PReluGradUpdateCompileInfo {};

IMPL_OP_OPTILING(PReluGradUpdate)
    .Tiling(PReluGradUpdateTilingFunc)
    .TilingParse<PReluGradUpdateCompileInfo>(TilingParseForPReluGradUpdate);

} // namespace optiling
