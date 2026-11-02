/*!
 * \file p_relu_grad_update_infershape.cpp
 * \brief Shape validation and inference for PReluGradUpdate.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {
namespace {

bool SameKnownShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        const int64_t l = lhs.GetDim(i);
        const int64_t r = rhs.GetDim(i);
        if (l >= 0 && r >= 0 && l != r) {
            return false;
        }
    }
    return true;
}

} // namespace

static ge::graphStatus InferShapePReluGradUpdate(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* grads = context->GetInputShape(0);
    const gert::Shape* features = context->GetInputShape(1);
    const gert::Shape* weights = context->GetInputShape(2);
    gert::Shape* dx = context->GetOutputShape(0);
    gert::Shape* update = context->GetOutputShape(1);
    if (grads == nullptr || features == nullptr || weights == nullptr || dx == nullptr || update == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (grads->GetDimNum() < 2 || !SameKnownShape(*grads, *features) || weights->GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t channel = grads->GetDim(1);
    const int64_t weightLen = weights->GetDim(0);
    if (channel >= 0 && weightLen >= 0 && weightLen != 1 && weightLen != channel) {
        return ge::GRAPH_FAILED;
    }

    *dx = *grads;
    *update = *features;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PReluGradUpdate).InferShape(InferShapePReluGradUpdate);

} // namespace ops
