/*!
 * \file p_relu_grad_update_tiling.cpp
 * \brief Runtime tiling for PReluGradUpdate.
 */

#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/p_relu_grad_update_tiling_data.h"
#include "../op_kernel/p_relu_grad_update_tiling_key.h"

namespace optiling {
namespace {

// The kernel has no GM scratch data. P026's temporary 16MB workspace was based
// on the old 507035 diagnosis and adds millisecond-scale fixed overhead.
constexpr uint32_t WS_SYS_SIZE = 0;
constexpr uint64_t UB_USAGE_NUMERATOR = 8U;
constexpr uint64_t UB_USAGE_DENOMINATOR = 10U;
// P100 每元素 UB（BUFFER_NUM=1）：
//   float: 4×4 + neg/fg(2×4) + mask≈1 = 25
//   fp16/bf16→fp32: 4×2 + float grads/features/dx/update/neg/fg(6×4) + mask≈1 = 33
constexpr int64_t FLOAT_BYTES_PER_ELEMENT = 25;
constexpr int64_t LOW_PRECISION_BYTES_PER_ELEMENT = 33;
constexpr int64_t CMP_ALIGN_FLOAT = 64;   // 中间计算在 fp32 上，Compare 按 64 齐

int64_t CeilDivI64(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int64_t AlignUpI64(int64_t value, int64_t alignment)
{
    return CeilDivI64(value, alignment) * alignment;
}

int64_t AlignDownI64(int64_t value, int64_t alignment)
{
    return (value / alignment) * alignment;
}

bool SameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context, "platform info is null");
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    // Atlas A2 上 Select 的 selMode=1/2 需要 8K Unified Buffer 作内部暂存，
    // 必须在 GetCoreMemSize 之前声明，否则返回的 ubSize 会把这 8K 算进可用空间。
    platform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_8K);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum <= 0 || ubSize == 0) {
        OP_LOGE(context, "invalid platform data: coreNum=%ld, ubSize=%lu", coreNum, ubSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        OP_LOGE(context, "workspace pointer is null");
        return ge::GRAPH_FAILED;
    }
    workspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

} // namespace

static ge::graphStatus PReluGradUpdateTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* gradsStorage = context->GetInputShape(0);
    const gert::StorageShape* featuresStorage = context->GetInputShape(1);
    const gert::StorageShape* weightsStorage = context->GetInputShape(2);
    const auto* inputDesc = context->GetInputDesc(0);
    if (gradsStorage == nullptr || featuresStorage == nullptr || weightsStorage == nullptr || inputDesc == nullptr) {
        OP_LOGE(context, "input shape or dtype is null");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& grads = gradsStorage->GetStorageShape();
    const gert::Shape& features = featuresStorage->GetStorageShape();
    const gert::Shape& weights = weightsStorage->GetStorageShape();
    if (grads.GetDimNum() < 2 || !SameShape(grads, features) || weights.GetDimNum() != 1) {
        OP_LOGE(context, "invalid ranks or mismatched grads/features shapes");
        return ge::GRAPH_FAILED;
    }

    const int64_t totalNum = grads.GetShapeSize();
    const int64_t channelNum = grads.GetDim(1);
    const int64_t weightLen = weights.GetDim(0);
    if (totalNum < 0 || channelNum <= 0 || (weightLen != 1 && weightLen != channelNum)) {
        OP_LOGE(context, "invalid shape values: total=%ld, channel=%ld, weightLen=%ld",
            totalNum, channelNum, weightLen);
        return ge::GRAPH_FAILED;
    }

    int64_t innerSize = 1;
    for (size_t i = 2; i < grads.GetDimNum(); ++i) {
        const int64_t dim = grads.GetDim(i);
        if (dim <= 0) {
            OP_LOGE(context, "all runtime dimensions must be positive");
            return ge::GRAPH_FAILED;
        }
        innerSize *= dim;
    }
    const int64_t sampleSize = channelNum * innerSize;

    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = 0;
    int64_t bytesPerElement = 0;
    int64_t cmpAlignElems = CMP_ALIGN_FLOAT;
    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        typeSize = 4;
        bytesPerElement = FLOAT_BYTES_PER_ELEMENT;
        cmpAlignElems = CMP_ALIGN_FLOAT;
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT);
#if PRELU_ENABLE_FP16
    } else if (dtype == ge::DT_FLOAT16) {
        typeSize = 2;
        bytesPerElement = LOW_PRECISION_BYTES_PER_ELEMENT;
        cmpAlignElems = CMP_ALIGN_FLOAT;
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16);
#endif
#if PRELU_ENABLE_BF16
    } else if (dtype == ge::DT_BF16) {
        typeSize = 2;
        bytesPerElement = LOW_PRECISION_BYTES_PER_ELEMENT;
        cmpAlignElems = CMP_ALIGN_FLOAT;
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16);
#endif
    } else {
        OP_LOGE(context, "unsupported dtype=%d", static_cast<int32_t>(dtype));
        return ge::GRAPH_FAILED;
    }

    const bool vectorWeights = weightLen > 1 && innerSize == 1;
    if (vectorWeights) {
        bytesPerElement += typeSize;  // contiguous channel weights queue
        if (dtype == ge::DT_FLOAT) {
            tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_S1);
        } else if (dtype == ge::DT_FLOAT16) {
            tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16_S1);
        } else {
            tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16_S1);
        }
    }

    if (!vectorWeights) {
        bytesPerElement -= sizeof(float);  // J008 featuresFloat reuse
    }

    uint64_t ubSize = 0;
    int64_t hardwareCoreNum = 0;
    if (GetPlatformInfo(context, ubSize, hardwareCoreNum) != ge::GRAPH_SUCCESS ||
        SetWorkspace(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t blockElements = 32 / typeSize;
    const uint64_t usableUb = ubSize * UB_USAGE_NUMERATOR / UB_USAGE_DENOMINATOR;
    int64_t maxTile = static_cast<int64_t>(usableUb / static_cast<uint64_t>(bytesPerElement));
    // kernel 会把 count 向上对齐到 cmpAlignElems；maxTile 取同倍数保证对齐后不超 UB。
    maxTile = AlignDownI64(maxTile, cmpAlignElems);
    if (maxTile < cmpAlignElems) {
        OP_LOGE(context, "UB is too small for one 256B-aligned tile");
        return ge::GRAPH_FAILED;
    }

    int64_t usedCores = 1;
    int64_t blockFactor = totalNum;
    int64_t tileLen = totalNum == 0 ? blockElements : std::min(totalNum, maxTile);
    int64_t rowsPerTile = 0;
    int64_t splitBatchGroups = 0;

    if (totalNum > 0) {
        const int64_t targetCoreNum = weightLen == 1
            ? std::min(hardwareCoreNum, std::max<int64_t>(4, CeilDivI64(totalNum, 1024)))
            : hardwareCoreNum;
        blockFactor = AlignUpI64(CeilDivI64(totalNum, targetCoreNum), blockElements);
        usedCores = std::min(targetCoreNum, CeilDivI64(totalNum, blockFactor));
        tileLen = std::min(blockFactor, maxTile);
        // A single weight is scalar even when C=1. Only distinct channel
        // weights use one-channel Muls tiles, or a contiguous C segment when S=1.
        if (weightLen > 1 && innerSize > 0) {
            if (!(vectorWeights && channelNum % blockElements == 0)) {
                tileLen = std::min(tileLen, vectorWeights ? channelNum : innerSize);
            }
        }
    }

    // Reuse the same weight phase across tiles without increasing loop count.
    if (vectorWeights && channelNum % blockElements == 0 && blockFactor > tileLen
        && tileLen > channelNum) {
        const int64_t residentTile = AlignDownI64(tileLen, channelNum);
        if (CeilDivI64(blockFactor, residentTile) == CeilDivI64(blockFactor, tileLen)) {
            tileLen = residentTile;
            rowsPerTile = residentTile / channelNum;
        }
    }

    // Separate binaries preserve the measured fallback instruction paths.
    if (vectorWeights && rowsPerTile > 0) {
        if (dtype == ge::DT_FLOAT) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_RESIDENT);
        else if (dtype == ge::DT_FLOAT16) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16_RESIDENT);
        else tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16_RESIDENT);
    } else if (vectorWeights && channelNum % blockElements == 0 && tileLen > channelNum) {
        if (dtype == ge::DT_FLOAT) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_MERGED);
        else if (dtype == ge::DT_FLOAT16) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16_MERGED);
        else tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16_MERGED);
    }

    // A core owns one channel tile and a contiguous batch range.
    // Reuse weights where a complete channel vector cannot fit in one tile.
    if (vectorWeights && channelNum % blockElements == 0 && channelNum > maxTile) {
        const int64_t cTiles = CeilDivI64(channelNum, maxTile);
        if (cTiles <= std::min<int64_t>(3, hardwareCoreNum)) {
            const int64_t batchGroups = hardwareCoreNum / cTiles;
            const int64_t batchSize = totalNum / channelNum;
            if (batchSize >= 2 * hardwareCoreNum) {
                tileLen = AlignUpI64(CeilDivI64(channelNum, cTiles), CMP_ALIGN_FLOAT);
                splitBatchGroups = batchGroups;
                usedCores = cTiles * batchGroups;
                if (dtype == ge::DT_FLOAT) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_SPLITC);
                else if (dtype == ge::DT_FLOAT16) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16_SPLITC);
                else tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16_SPLITC);
            }
        }
    }

    PReluGradUpdateTilingData* tiling = context->GetTilingData<PReluGradUpdateTilingData>();
    if (tiling == nullptr) {
        OP_LOGE(context, "tiling data pointer is null");
        return ge::GRAPH_FAILED;
    }
    tiling->totalNum = totalNum;
    tiling->coreNum = usedCores;
    tiling->blockFactor = blockFactor;
    tiling->tileLen = tileLen;
    tiling->channelNum = channelNum;
    tiling->innerSize = innerSize;
    tiling->sampleSize = sampleSize;
    tiling->weightsPerChannel = weightLen > 1 ? 1 : 0;
    tiling->rowsPerTile = rowsPerTile;
    tiling->splitBatchGroups = splitBatchGroups;

    context->SetBlockDim(static_cast<uint32_t>(usedCores));
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
