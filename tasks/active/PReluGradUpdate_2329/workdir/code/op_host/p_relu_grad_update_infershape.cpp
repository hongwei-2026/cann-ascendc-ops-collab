/*!
 * \file p_relu_grad_update_infershape.cpp
 * \brief InferShape + checks (design_final §3.2.2)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static bool HasUnknownDim(const gert::Shape* s)
{
    if (s == nullptr) {
        return true;
    }
    for (size_t i = 0; i < s->GetDimNum(); ++i) {
        if (s->GetDim(i) < 0) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus InferShapePReluGradUpdate(gert::InferShapeContext* context)
{
    const gert::Shape* grads = context->GetInputShape(0);
    const gert::Shape* features = context->GetInputShape(1);
    const gert::Shape* weights = context->GetInputShape(2);
    gert::Shape* dx = context->GetOutputShape(0);
    gert::Shape* update = context->GetOutputShape(1);
    if (grads == nullptr || features == nullptr || weights == nullptr || dx == nullptr || update == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (grads->GetDimNum() < 2 || features->GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    if (weights->GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const bool dynamic = HasUnknownDim(grads) || HasUnknownDim(features) || HasUnknownDim(weights);
    if (!dynamic) {
        if (*grads != *features) {
            return ge::GRAPH_FAILED;
        }
        const int64_t c = grads->GetDim(1);
        const int64_t wLen = weights->GetDim(0);
        if (wLen != 1 && wLen != c) {
            return ge::GRAPH_FAILED;
        }
    }

    *dx = *grads;
    *update = *features;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PReluGradUpdate).InferShape(InferShapePReluGradUpdate);

} // namespace ops
