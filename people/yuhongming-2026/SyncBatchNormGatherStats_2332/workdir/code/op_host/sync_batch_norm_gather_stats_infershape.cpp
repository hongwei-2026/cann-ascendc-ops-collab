/*!
 * \file sync_batch_norm_gather_stats_infershape.cpp
 * \brief SyncBatchNormGatherStats 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeSyncBatchNormGatherStats(gert::InferShapeContext* context)
{
    // batch_mean / batch_invstd / new_mean / new_variance 都是逐通道，形状与 running mean 一致。
    const gert::Shape* meanShape = context->GetInputShape(3);
    if (meanShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 4; ++i) {
        gert::Shape* outputShape = context->GetOutputShape(i);
        if (outputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *outputShape = *meanShape;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SyncBatchNormGatherStats).InferShape(InferShapeSyncBatchNormGatherStats);

} // namespace ops
