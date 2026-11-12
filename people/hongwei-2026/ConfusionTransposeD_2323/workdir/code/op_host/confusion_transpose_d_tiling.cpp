/*!
 * \file confusion_transpose_d_tiling.cpp
 * \brief ConfusionTransposeD 算子 Tiling 实现
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/confusion_transpose_d_tiling_data.h"
#include "../op_kernel/confusion_transpose_d_tiling_key.h"

// Platform packages disallow new Host files; keep CTD_PROBE inlined here (and in infershape).
#ifndef CONFUSION_TRANSPOSE_D_PROBE_H
#define CONFUSION_TRANSPOSE_D_PROBE_H
namespace ctd_probe {
inline uint64_t NextId()
{
    static std::atomic<uint64_t> sequence{0};
    return ++sequence;
}
template <typename Shape>
void WriteShape(std::ostream& out, const Shape* shape)
{
    if (shape == nullptr) {
        out << "null";
        return;
    }
    out << '[';
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        if (i != 0) out << ',';
        out << shape->GetDim(i);
    }
    out << ']';
}
template <typename Vector>
void WriteVector(std::ostream& out, const Vector* vec)
{
    if (vec == nullptr) {
        out << "null";
        return;
    }
    const auto* data = reinterpret_cast<const int64_t*>(vec->GetData());
    if (data == nullptr && vec->GetSize() != 0) {
        out << "null";
        return;
    }
    out << '[';
    for (size_t i = 0; i < vec->GetSize(); ++i) {
        if (i != 0) out << ',';
        out << data[i];
    }
    out << ']';
}
template <typename Shape>
void WriteNumel(std::ostream& out, const Shape* shape)
{
    if (shape == nullptr) {
        out << "null";
        return;
    }
    int64_t value = 1;
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        const int64_t dim = shape->GetDim(i);
        if (dim < 0 || (dim != 0 && value > std::numeric_limits<int64_t>::max() / dim)) {
            out << "null";
            return;
        }
        value *= dim;
    }
    out << value;
}
class Event {
public:
    explicit Event(const char* phase) : phase_(phase), id_(NextId()) {}
    ~Event() { Emit("exit"); }
    template <typename Shape, typename Attrs>
    void Capture(const Shape* input, const Attrs* attrs, int dtype)
    {
        std::ostringstream out;
        out << ",\"dtype_code\":";
        if (dtype < 0) out << "null"; else out << dtype;
        out << ",\"in_shape\":";
        WriteShape(out, input);
        out << ",\"numel\":";
        WriteNumel(out, input);
        out << ",\"perm\":";
        const auto* perm = attrs ? attrs->template GetAttrPointer<gert::ContinuousVector>(0) : nullptr;
        const auto* shape = attrs ? attrs->template GetAttrPointer<gert::ContinuousVector>(1) : nullptr;
        const bool* tf = attrs ? attrs->template GetAttrPointer<bool>(2) : nullptr;
        WriteVector(out, perm);
        out << ",\"attr_shape\":";
        WriteVector(out, shape);
        out << ",\"transpose_first\":";
        if (tf == nullptr) out << "null"; else out << (*tf ? "true" : "false");
        fields_ = out.str();
        Emit("enter");
    }
    template <typename Shape>
    void Output(const Shape* shape)
    {
        std::ostringstream out;
        out << ",\"out_shape\":";
        WriteShape(out, shape);
        extra_ += out.str();
    }
    template <typename Shape>
    void OutputStorage(const Shape* shape)
    {
        std::ostringstream out;
        out << ",\"tiling_out_storage_shape\":";
        WriteShape(out, shape);
        extra_ += out.str();
    }
    void Limits(uint64_t ub, int64_t numel, int64_t typeSize)
    {
        std::ostringstream out;
        out << ",\"ub_bytes\":" << ub << ",\"element_bytes\":" << typeSize << ",\"input_bytes\":";
        if (numel >= 0 && typeSize > 0 && numel <= std::numeric_limits<int64_t>::max() / typeSize) {
            out << numel * typeSize;
        } else {
            out << "null";
        }
        extra_ += out.str();
    }
    void Tiling(int64_t span, int64_t cores, int64_t block, int64_t tile)
    {
        std::ostringstream out;
        out << ",\"copy_span\":" << span << ",\"path_hint\":\""
            << (span > 1 ? "contiguous_row" : "scalar") << "\",\"block_dim\":" << cores
            << ",\"block_factor\":" << block << ",\"ub_factor\":" << tile;
        extra_ += out.str();
    }
    void Stage(const char* stage) { stage_ = stage; }
    void Success() { success_ = true; stage_ = "complete"; }
private:
    void Emit(const char* event) const
    {
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::ostringstream out;
        out << "CTD_PROBE {\"pid\":" << static_cast<long>(getpid()) << ",\"seq\":" << id_
            << ",\"time_unix_us\":" << us << ",\"phase\":\"" << phase_ << "\",\"event\":\""
            << event << "\",\"stage\":\"" << stage_ << "\",\"host_ok\":";
        if (event[1] == 'n') out << "null"; else out << (success_ ? "true" : "false");
        out << fields_ << extra_ << '}';
        const auto line = out.str();
        std::fprintf(stderr, "%s\n", line.c_str());
        std::fflush(stderr);
    }
    const char* phase_;
    uint64_t id_;
    const char* stage_ = "entry";
    bool success_ = false;
    std::string fields_;
    std::string extra_;
};
} // namespace ctd_probe
#endif

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr int64_t MAX_DIM = CTD_MAX_DIM;

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

static int64_t GetTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return 4;
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return 8;
        default:
            return 4;
    }
}

static void SetDimField(ConfusionTransposeDTilingData* tiling, int64_t* dims, int64_t* shape, int64_t* perm)
{
    tiling->dimIn0 = dims[0]; tiling->dimIn1 = dims[1]; tiling->dimIn2 = dims[2]; tiling->dimIn3 = dims[3];
    tiling->dimIn4 = dims[4]; tiling->dimIn5 = dims[5]; tiling->dimIn6 = dims[6]; tiling->dimIn7 = dims[7];
    tiling->dimShape0 = shape[0]; tiling->dimShape1 = shape[1]; tiling->dimShape2 = shape[2]; tiling->dimShape3 = shape[3];
    tiling->dimShape4 = shape[4]; tiling->dimShape5 = shape[5]; tiling->dimShape6 = shape[6]; tiling->dimShape7 = shape[7];
    tiling->perm0 = perm[0]; tiling->perm1 = perm[1]; tiling->perm2 = perm[2]; tiling->perm3 = perm[3];
    tiling->perm4 = perm[4]; tiling->perm5 = perm[5]; tiling->perm6 = perm[6]; tiling->perm7 = perm[7];
}

static int64_t GetCopySpan(const int64_t* dims, const int64_t* perm, int64_t rank)
{
    int64_t strides[MAX_DIM] = {0};
    strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    int64_t span = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
        const int64_t axis = perm[i];
        if (dims[axis] == 1) {
            continue;
        }
        if (strides[axis] != span) {
            break;
        }
        span *= dims[axis];
    }
    return span;
}

static ge::graphStatus ConfusionTransposeDTilingFunc(gert::TilingContext* context)
{
    ctd_probe::Event probe("tiling");
    const auto* probeStorage = context->GetInputShape(0);
    const auto* probeDesc = context->GetInputDesc(0);
    const auto* probeShape = probeStorage ? &probeStorage->GetStorageShape() : nullptr;
    probe.Capture(probeShape, context->GetAttrs(), probeDesc ? static_cast<int>(probeDesc->GetDataType()) : -1);
    const auto* probeOutput = context->GetOutputShape(0);
    probe.OutputStorage(probeOutput ? &probeOutput->GetStorageShape() : nullptr);
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    probe.Stage("platform_info");
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    probe.Stage("workspace");
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* xStorage = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    probe.Stage("missing_input_shape_or_desc");
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const gert::Shape& xShape = xStorage->GetStorageShape();
    const size_t rankIn = xShape.GetDimNum();
    probe.Stage("invalid_input_rank");
    OP_CHECK_IF(rankIn == 0 || rankIn > MAX_DIM, OP_LOGE(context, "invalid input rank"), return ge::GRAPH_FAILED);

    int64_t dimIn[MAX_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t totalNum = 1;
    for (size_t i = 0; i < rankIn; ++i) {
        dimIn[i] = xShape.GetDim(i);
        totalNum *= dimIn[i];
    }
    probe.Stage("empty_or_invalid_numel");
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "empty tensor"), return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    probe.Stage("missing_attributes");
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* permVec = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const auto* shapeVec = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const bool* transposeFirst = attrs->GetAttrPointer<bool>(2);
    probe.Stage("missing_attribute_value");
    OP_CHECK_NULL_WITH_CONTEXT(context, permVec);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeVec);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeFirst);

    const size_t rank = shapeVec->GetSize();
    probe.Stage("perm_shape_rank_mismatch");
    OP_CHECK_IF(
        rank == 0 || rank > MAX_DIM || permVec->GetSize() != rank,
        OP_LOGE(context, "perm/shape rank mismatch"),
        return ge::GRAPH_FAILED);

    const int64_t* permData = reinterpret_cast<const int64_t*>(permVec->GetData());
    const int64_t* shapeData = reinterpret_cast<const int64_t*>(shapeVec->GetData());
    int64_t dimShape[MAX_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t perm[MAX_DIM] = {0, 1, 2, 3, 4, 5, 6, 7};
    int64_t shapeNumel = 1;
    for (size_t i = 0; i < rank; ++i) {
        dimShape[i] = shapeData[i];
        perm[i] = permData[i];
        shapeNumel *= dimShape[i];
        probe.Stage("perm_out_of_range");
        OP_CHECK_IF(
            perm[i] < 0 || perm[i] >= static_cast<int64_t>(rank),
            OP_LOGE(context, "perm out of range"),
            return ge::GRAPH_FAILED);
    }
    probe.Stage("shape_numel_mismatch");
    OP_CHECK_IF(shapeNumel != totalNum, OP_LOGE(context, "shape numel mismatch"), return ge::GRAPH_FAILED);

    const int64_t typeSize = GetTypeSize(inputDesc->GetDataType());
    // 输入整 tensor 常驻 UB + 输出 tile；预留 8KB
    const int64_t reserved = 8 * 1024;
    probe.Limits(ubSize, totalNum, typeSize);
    probe.Stage("tensor_too_large_for_ub");
    OP_CHECK_IF(
        totalNum * typeSize + reserved >= static_cast<int64_t>(ubSize),
        OP_LOGE(context, "tensor too large for UB-resident scheme"),
        return ge::GRAPH_FAILED);

    const int64_t remainUb = static_cast<int64_t>(ubSize) - totalNum * typeSize - reserved;
    int64_t ubFactor = remainUb / typeSize;
    if (ubFactor > totalNum) {
        ubFactor = totalNum;
    }
    if (ubFactor <= 0) {
        ubFactor = 1;
    }
    // 32B 对齐友好：向下对齐到 32/typeSize
    const int64_t align = 32 / typeSize;
    if (align > 1 && ubFactor >= align) {
        ubFactor = (ubFactor / align) * align;
    }

    int64_t usedCores = coreNum;
    if (usedCores > totalNum) {
        usedCores = totalNum;
    }
    if (usedCores <= 0) {
        usedCores = 1;
    }
    int64_t blockFactor = (totalNum + usedCores - 1) / usedCores;
    // TF=false 的转置输入是 attr shape；TF=true 是原输入。
    const int64_t copySpan = (*transposeFirst && rankIn != rank) ? 1 :
        GetCopySpan((*transposeFirst) ? dimIn : dimShape, perm, static_cast<int64_t>(rank));
    if (copySpan > 1) {
        // 核边界同时对齐整行和 32B，避免相邻核写同一 GM 块。
        int64_t a = copySpan;
        int64_t b = align;
        while (b != 0) {
            const int64_t remainder = a % b;
            a = b;
            b = remainder;
        }
        const int64_t grain = copySpan * (align / a);
        const int64_t grains = (totalNum + grain - 1) / grain;
        usedCores = std::min(coreNum, grains);
        blockFactor = ((grains + usedCores - 1) / usedCores) * grain;
        usedCores = (totalNum + blockFactor - 1) / blockFactor;
    }

    ConfusionTransposeDTilingData* tiling = context->GetTilingData<ConfusionTransposeDTilingData>();
    probe.Stage("missing_tiling_data");
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->transposeFirst = (*transposeFirst) ? 1 : 0;
    tiling->rank = static_cast<int64_t>(rank);
    tiling->rankIn = static_cast<int64_t>(rankIn);
    tiling->copySpan = copySpan;
    SetDimField(tiling, dimIn, dimShape, perm);

    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    uint64_t tilingKey = 0;
    if (typeSize == 2) {
        tilingKey = GET_TPL_TILING_KEY(CONFUSIONTRANSPOSED_TPL_SCH_MODE_0);
    } else if (typeSize == 4) {
        tilingKey = GET_TPL_TILING_KEY(CONFUSIONTRANSPOSED_TPL_SCH_MODE_1);
    } else if (typeSize == 1) {
        tilingKey = GET_TPL_TILING_KEY(CONFUSIONTRANSPOSED_TPL_SCH_MODE_2);
    } else {
        tilingKey = GET_TPL_TILING_KEY(CONFUSIONTRANSPOSED_TPL_SCH_MODE_3);
    }
    context->SetTilingKey(tilingKey);
    probe.Tiling(copySpan, usedCores, blockFactor, ubFactor);
    probe.Success();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForConfusionTransposeD([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ConfusionTransposeDCompileInfo {};

IMPL_OP_OPTILING(ConfusionTransposeD)
    .Tiling(ConfusionTransposeDTilingFunc)
    .TilingParse<ConfusionTransposeDCompileInfo>(TilingParseForConfusionTransposeD);

} // namespace optiling
