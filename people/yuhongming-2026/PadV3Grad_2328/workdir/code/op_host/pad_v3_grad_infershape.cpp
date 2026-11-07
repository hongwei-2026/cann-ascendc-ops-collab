/*!
 * \file pad_v3_grad_infershape.cpp
 * \brief y 的每一维 = x 的该维减去 front 和 back。
 */

#include "pad_v3_grad_parse.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapePadV3Grad(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    if (rank == 0) {
        yShape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }
    if (rank > kPadV3GradMaxRank) {
        return ge::GRAPH_FAILED;
    }

    bool contiguous = true;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* flag = attrs->GetAttrPointer<bool>(1);
        if (flag != nullptr) {
            contiguous = *flag;
        }
    }

    const gert::Tensor* pads = context->GetInputTensor(1);
    if (pads == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t front[kPadV3GradMaxRank] = {};
    int64_t back[kPadV3GradMaxRank] = {};
    const int64_t count = pads->GetShapeSize();
    bool ok = false;
    if (pads->GetDataType() == ge::DT_INT64) {
        ok = PadV3GradReadPads(pads->GetData<int64_t>(), count, rank, contiguous, front, back);
    } else {
        ok = PadV3GradReadPads(pads->GetData<int32_t>(), count, rank, contiguous, front, back);
    }
    if (!ok) {
        return ge::GRAPH_FAILED;
    }

    yShape->SetDimNum(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t dim = xShape->GetDim(static_cast<size_t>(i));
        if (dim < 0) {
            yShape->SetDim(static_cast<size_t>(i), -1);
            continue;
        }
        const int64_t out = dim - front[i] - back[i];
        if (out < 0) {
            return ge::GRAPH_FAILED;
        }
        yShape->SetDim(static_cast<size_t>(i), out);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PadV3Grad).InferShape(InferShapePadV3Grad);

} // namespace ops
