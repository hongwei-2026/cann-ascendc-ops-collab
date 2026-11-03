/*!
 * \file multilabel_margin_loss_tiling.cpp
 * \brief MultilabelMarginLoss 算子 Tiling 实现
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/multilabel_margin_loss_tiling_data.h"
#include "../op_kernel/multilabel_margin_loss_tiling_key.h"

namespace optiling {
namespace {

constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr int64_t REDUCTION_NONE = 0;
constexpr int64_t REDUCTION_MEAN = 1;
constexpr int64_t REDUCTION_SUM = 2;
constexpr int64_t REDUCTION_INVALID = -1;

int64_t CeilDivI64(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int64_t ParseReductionAttr(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return REDUCTION_MEAN;
    }
    const char* reduction = attrs->GetAttrPointer<char>(0);
    if (reduction == nullptr) {
        return REDUCTION_MEAN;
    }
    if (std::strcmp(reduction, "none") == 0) {
        return REDUCTION_NONE;
    }
    if (std::strcmp(reduction, "mean") == 0) {
        return REDUCTION_MEAN;
    }
    if (std::strcmp(reduction, "sum") == 0) {
        return REDUCTION_SUM;
    }
    return REDUCTION_INVALID;
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

ge::graphStatus SetWorkspace(gert::TilingContext* context, int64_t batchSize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    // 每行一份 float32 中间 loss，供多核归约；外加系统预留。
    const size_t rowLossBytes = static_cast<size_t>(std::max<int64_t>(batchSize, 1)) * sizeof(float);
    currentWorkspace[0] = WS_SYS_SIZE + rowLossBytes;
    return ge::GRAPH_SUCCESS;
}

} // namespace

static ge::graphStatus MultilabelMarginLossTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* xStorage = context->GetInputShape(0);
    const gert::StorageShape* targetStorage = context->GetInputShape(1);
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const gert::Shape& xShape = xStorage->GetStorageShape();
    const gert::Shape& targetShape = targetStorage->GetStorageShape();
    const size_t xRank = xShape.GetDimNum();
    const size_t tRank = targetShape.GetDimNum();
    OP_CHECK_IF(
        xRank == 0 || xRank > 2 || tRank == 0 || tRank > 2,
        OP_LOGE(context, "x/target rank must be 1 or 2"),
        return ge::GRAPH_FAILED);

    int64_t batchSize = 1;
    int64_t classNum = 0;
    if (xRank == 1) {
        classNum = xShape.GetDim(0);
    } else {
        batchSize = xShape.GetDim(0);
        classNum = xShape.GetDim(1);
    }
    int64_t targetLen = 0;
    if (tRank == 1) {
        targetLen = targetShape.GetDim(0);
        OP_CHECK_IF(
            xRank == 2,
            OP_LOGE(context, "target rank 1 only valid when x is 1D"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            targetShape.GetDim(0) != batchSize,
            OP_LOGE(context, "target batch mismatch"),
            return ge::GRAPH_FAILED);
        targetLen = targetShape.GetDim(1);
    }
    OP_CHECK_IF(
        batchSize <= 0 || classNum <= 0 || targetLen <= 0,
        OP_LOGE(context, "invalid N/C/K: %ld/%ld/%ld", batchSize, classNum, targetLen),
        return ge::GRAPH_FAILED);

    int64_t xTypeSize = 4;
    const ge::DataType dtypeProbe = inputDesc->GetDataType();
    if (dtypeProbe == ge::DT_FLOAT16 || dtypeProbe == ge::DT_BF16) {
        xTypeSize = 2;
    } else if (dtypeProbe != ge::DT_FLOAT) {
        OP_LOGE(context, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }
    // UB 粗估：x/target/is_target/float 工作缓冲；一行必须能放下。
    const int64_t xBytes = classNum * xTypeSize;
    const int64_t targetBytes = targetLen * static_cast<int64_t>(sizeof(int32_t));
    const int64_t isTargetBytes = classNum * static_cast<int64_t>(sizeof(int32_t));
    const int64_t floatWorkBytes = classNum * static_cast<int64_t>(sizeof(float)) * 3; // xFloat + scratch
    const int64_t needUb = xBytes + targetBytes + isTargetBytes + floatWorkBytes + 4096;
    OP_CHECK_IF(
        static_cast<uint64_t>(needUb) > ubSize,
        OP_LOGE(context, "row footprint %ld exceeds UB %lu", needUb, ubSize),
        return ge::GRAPH_FAILED);

    const int64_t reduction = ParseReductionAttr(context);
    OP_CHECK_IF(
        reduction == REDUCTION_INVALID,
        OP_LOGE(context, "reduction must be none/mean/sum"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetWorkspace(context, batchSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    MultilabelMarginLossTilingData* tiling = context->GetTilingData<MultilabelMarginLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const int64_t usedCores = std::min(coreNum, batchSize);
    const int64_t rowsPerCore = CeilDivI64(batchSize, usedCores);
    const int64_t fullCores = (rowsPerCore == 0) ? 0 : (batchSize / rowsPerCore);
    const int64_t rem = (rowsPerCore == 0) ? 0 : (batchSize % rowsPerCore);
    // 采用均匀前缀分核：核 i 处理 [i*rowsPerCore, min((i+1)*rowsPerCore, N))
    tiling->batchSize = batchSize;
    tiling->classNum = classNum;
    tiling->targetLen = targetLen;
    tiling->reduction = reduction;
    tiling->usedCores = usedCores;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tailRows = rem; // 仅记录；kernel 用 min 截断
    (void)fullCores;

    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    uint64_t tilingKey = GET_TPL_TILING_KEY(MULTILABELMARGINLOSS_TPL_SCH_MODE_1);
    const ge::DataType dtype = inputDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(MULTILABELMARGINLOSS_TPL_SCH_MODE_0);
    } else if (dtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(MULTILABELMARGINLOSS_TPL_SCH_MODE_1);
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(MULTILABELMARGINLOSS_TPL_SCH_MODE_2);
    } else {
        OP_LOGE(context, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMultilabelMarginLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MultilabelMarginLossCompileInfo {};

IMPL_OP_OPTILING(MultilabelMarginLoss)
    .Tiling(MultilabelMarginLossTilingFunc)
    .TilingParse<MultilabelMarginLossCompileInfo>(TilingParseForMultilabelMarginLoss);

} // namespace optiling
