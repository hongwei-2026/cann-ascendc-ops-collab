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
// P104 每元素 UB（BUFFER_NUM=1，含每核常驻 zero buffer）：
//   float: 4×4 + neg/fg/zero(3×4) + mask≈1 = 29
//   fp16/bf16→fp32: 4×2 + float grads/features/dx/update/neg/fg/zero(7×4) + mask≈1 = 37
constexpr int64_t FLOAT_BYTES_PER_ELEMENT = 29;
constexpr int64_t LOW_PRECISION_BYTES_PER_ELEMENT = 37;
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

    if (totalNum > 0) {
        blockFactor = AlignUpI64(CeilDivI64(totalNum, hardwareCoreNum), blockElements);
        usedCores = std::min(hardwareCoreNum, CeilDivI64(totalNum, blockFactor));
        tileLen = std::min(blockFactor, maxTile);
        // When weights are [C], one Compute uses a single Muls weight — keep the
        // host tile inside one channel (kernel also clamps and steps by currentNum).
        if (weightLen == channelNum && innerSize > 0) {
            tileLen = std::min(tileLen, innerSize);
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
    tiling->weightsPerChannel = weightLen == channelNum ? 1 : 0;
    tiling->rowsPerTile = rowsPerTile;

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
