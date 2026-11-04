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

ge::graphStatus SetWorkspace(gert::TilingContext* context, int64_t batchSize,
    int64_t reduction, int64_t usedCores)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    // Only multi-core mean/sum consume GM scratch. SyncAll() is the hardware
    // overload; no Matmul/HCCL or software-sync API consumes system workspace.
    currentWorkspace[0] = reduction == REDUCTION_NONE || usedCores == 1
        ? 0 : static_cast<size_t>(CeilDivI64(batchSize, 8)) * 32;
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
    // UB：优先整行；装不下则按 design_final 滑窗切 C（target 仍整行，避免拆截断前缀）。
    // reserve 覆盖 kernel 的 redBuf_(RED_CHUNK=512 float=2KB)、outBuf_ 与 UB 预留。
    const int64_t reserve = 12288;
    auto Footprint = [&](int64_t tileC, int64_t tileK) -> int64_t {
        const int64_t xRawBytes = tileC * xTypeSize;                 // xRawBuf_
        const int64_t targetBytes = tileK * static_cast<int64_t>(sizeof(int32_t)); // tgtBuf_
        const int64_t cntBytes = tileC * static_cast<int64_t>(sizeof(int32_t));    // cntBuf_
        // xjfBuf_ / xlfBuf_ / maskBuf_ / accBuf_ / tmpBuf_ 共 5 份 float
        const int64_t floatWorkBytes = tileC * static_cast<int64_t>(sizeof(float)) * 5;
        return xRawBytes + targetBytes + cntBytes + floatWorkBytes + reserve;
    };
    int64_t classTile = classNum;
    int64_t targetTile = targetLen;
    if (static_cast<uint64_t>(Footprint(classTile, targetTile)) > ubSize) {
        OP_CHECK_IF(
            static_cast<uint64_t>(Footprint(8, targetLen)) > ubSize,
            OP_LOGE(context, "targetLen=%ld too large for UB %lu", targetLen, ubSize),
            return ge::GRAPH_FAILED);
        int64_t lo = 8;
        int64_t hi = classNum;
        classTile = 8;
        while (lo <= hi) {
            const int64_t mid = (lo + hi) / 2;
            if (static_cast<uint64_t>(Footprint(mid, targetLen)) <= ubSize) {
                classTile = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        classTile = (classTile / 8) * 8;
        OP_CHECK_IF(
            classTile < 8,
            OP_LOGE(context, "cannot find classTile for C=%ld UB=%lu", classNum, ubSize),
            return ge::GRAPH_FAILED);
    }

    const int64_t reduction = ParseReductionAttr(context);
    OP_CHECK_IF(
        reduction == REDUCTION_INVALID,
        OP_LOGE(context, "reduction must be none/mean/sum"),
        return ge::GRAPH_FAILED);

    MultilabelMarginLossTilingData* tiling = context->GetTilingData<MultilabelMarginLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 分核：先按核数上限定每核行数，再回算**实际有活**的核数。
    // 若直接用 min(coreNum,N) 当 blockDim，当 ceil(N/rowsPerCore) < min(coreNum,N) 时
    // （如 N=100、coreNum=64 → rowsPerCore=2，只有 50 核有活）会拉起 14 个空闲核；
    // 这些核在 kernel 里拿不到行，一旦提前退出就缺席 SyncAll，使其余核永久等待。
    // Bound the single-core path by worst-case pair work, independently of
    // target contents. Larger reductions retain the original row parallelism.
    const bool tinyReduction = reduction != REDUCTION_NONE && batchSize <= 8 &&
        classNum <= 32 && targetLen <= 32 && batchSize * classNum * targetLen <= 4096;
    const int64_t coresCap = tinyReduction ? 1 : std::min(coreNum, batchSize);
    const int64_t rowsPerCore = CeilDivI64(batchSize, coresCap);
    const int64_t usedCores = CeilDivI64(batchSize, rowsPerCore);
    const int64_t rem = batchSize - (usedCores - 1) * rowsPerCore; // 尾核实际行数
    OP_CHECK_IF(
        SetWorkspace(context, batchSize, reduction, usedCores) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    // 采用均匀前缀分核：核 i 处理 [i*rowsPerCore, min((i+1)*rowsPerCore, N))
    tiling->batchSize = batchSize;
    tiling->classNum = classNum;
    tiling->targetLen = targetLen;
    tiling->reduction = reduction;
    tiling->usedCores = usedCores;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tailRows = rem; // 仅记录；kernel 用 min 截断
    tiling->classTile = classTile;
    tiling->targetTile = targetTile;

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
