/*!
 * \file ctc_loss_v2_tiling.cpp
 * \brief CtcLossV2 Tiling（按 batch 分核；样例 N=1）
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/ctc_loss_v2_tiling_data.h"
#include "../op_kernel/ctc_loss_v2_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

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

static ge::graphStatus CtcLossV2TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = WS_SYS_SIZE;

    const gert::StorageShape* lpStorage = context->GetInputShape(0);
    const gert::StorageShape* tgStorage = context->GetInputShape(1);
    const auto* lpDesc = context->GetInputDesc(0);
    const auto* tgDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, lpStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, tgStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, lpDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, tgDesc);

    const gert::Shape& lp = lpStorage->GetStorageShape();
    const gert::Shape& tg = tgStorage->GetStorageShape();
    OP_CHECK_IF(lp.GetDimNum() != 3, OP_LOGE(context, "log_probs must be 3D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tg.GetDimNum() == 0, OP_LOGE(context, "targets rank 0"), return ge::GRAPH_FAILED);

    const int64_t T = lp.GetDim(0);
    const int64_t N = lp.GetDim(1);
    const int64_t C = lp.GetDim(2);
    const int64_t S = tg.GetDim(tg.GetDimNum() - 1);
    OP_CHECK_IF(T <= 0 || N <= 0 || C <= 0 || S < 0, OP_LOGE(context, "invalid dims"), return ge::GRAPH_FAILED);

    int64_t blank = 0;
    int64_t zeroInfinity = 0;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* blankPtr = attrs->GetAttrPointer<int64_t>(0);
        if (blankPtr != nullptr) {
            blank = *blankPtr;
        }
        const bool* ziPtr = attrs->GetAttrPointer<bool>(2);
        if (ziPtr != nullptr) {
            zeroInfinity = (*ziPtr) ? 1 : 0;
        }
    }

    int64_t usedCores = std::min(coreNum, N);
    if (usedCores <= 0) {
        usedCores = 1;
    }

    CtcLossV2TilingData* tiling = context->GetTilingData<CtcLossV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->timeSize = T;
    tiling->batchSize = N;
    tiling->classNum = C;
    tiling->maxTargetLen = S;
    tiling->blank = blank;
    tiling->zeroInfinity = zeroInfinity;
    tiling->targetsIsInt64 = (tgDesc->GetDataType() == ge::DT_INT64) ? 1 : 0;
    tiling->usedCores = usedCores;

    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    // def 仅 float log_probs → mode 1；half 预留 mode 0
    uint64_t tilingKey = GET_TPL_TILING_KEY(CTCLOSSV2_TPL_SCH_MODE_1);
    if (lpDesc->GetDataType() == ge::DT_FLOAT16 || lpDesc->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(CTCLOSSV2_TPL_SCH_MODE_0);
    }
    context->SetTilingKey(tilingKey);
    (void)ubSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCtcLossV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CtcLossV2CompileInfo {};

IMPL_OP_OPTILING(CtcLossV2)
    .Tiling(CtcLossV2TilingFunc)
    .TilingParse<CtcLossV2CompileInfo>(TilingParseForCtcLossV2);

} // namespace optiling
