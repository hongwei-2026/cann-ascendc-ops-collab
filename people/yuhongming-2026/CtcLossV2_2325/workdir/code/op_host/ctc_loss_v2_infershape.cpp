/*!
 * \file ctc_loss_v2_infershape.cpp
 * \brief CtcLossV2 形状推导：loss[N], log_alpha[N,T,2*S+1]
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeCtcLossV2(gert::InferShapeContext* context)
{
    const gert::Shape* logProbsShape = context->GetInputShape(0);
    const gert::Shape* targetsShape = context->GetInputShape(1);
    gert::Shape* lossShape = context->GetOutputShape(0);
    gert::Shape* alphaShape = context->GetOutputShape(1);
    if (logProbsShape == nullptr || targetsShape == nullptr || lossShape == nullptr || alphaShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (logProbsShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    const int64_t T = logProbsShape->GetDim(0);
    const int64_t N = logProbsShape->GetDim(1);
    const size_t tRank = targetsShape->GetDimNum();
    if (tRank == 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t S = targetsShape->GetDim(tRank - 1);
    const int64_t U = 2 * S + 1;

    lossShape->SetDimNum(1);
    lossShape->SetDim(0, N);

    alphaShape->SetDimNum(3);
    alphaShape->SetDim(0, N);
    alphaShape->SetDim(1, T);
    alphaShape->SetDim(2, U);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CtcLossV2).InferShape(InferShapeCtcLossV2);

} // namespace ops
