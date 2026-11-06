/*!
 * \file layer_norm_v3_infershape.cpp
 * \brief y 与 x 同形；mean/rstd 从 begin_norm_axis 起各维为 1。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static int64_t NormalizeAxis(int64_t axis, int64_t rank)
{
    if (axis < 0) {
        axis += rank;
    }
    return axis;
}

static ge::graphStatus InferShapeLayerNormV3(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;

    const int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    int64_t beginNorm = 0;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* axis = attrs->GetAttrPointer<int64_t>(0);
        if (axis != nullptr) {
            beginNorm = *axis;
        }
    }

    gert::Shape* meanShape = context->GetOutputShape(1);
    gert::Shape* rstdShape = context->GetOutputShape(2);
    if (rank == 0) {
        if (meanShape != nullptr) {
            *meanShape = *xShape;
        }
        if (rstdShape != nullptr) {
            *rstdShape = *xShape;
        }
        return ge::GRAPH_SUCCESS;
    }

    beginNorm = NormalizeAxis(beginNorm, rank);
    if (beginNorm < 0 || beginNorm > rank) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape reduced = *xShape;
    for (int64_t i = beginNorm; i < rank; ++i) {
        reduced.SetDim(static_cast<size_t>(i), 1);
    }
    if (meanShape != nullptr) {
        *meanShape = reduced;
    }
    if (rstdShape != nullptr) {
        *rstdShape = reduced;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LayerNormV3).InferShape(InferShapeLayerNormV3);

} // namespace ops
