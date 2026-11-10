/*!
 * \file confusion_transpose_d_infershape.cpp
 * \brief ConfusionTransposeD 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeConfusionTransposeD(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* perm = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const auto* shapeAttr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const bool* transposeFirst = attrs->GetAttrPointer<bool>(2);
    if (perm == nullptr || shapeAttr == nullptr || transposeFirst == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t* permData = reinterpret_cast<const int64_t*>(perm->GetData());
    const int64_t* shapeData = reinterpret_cast<const int64_t*>(shapeAttr->GetData());
    const size_t rank = shapeAttr->GetSize();
    if (rank == 0 || perm->GetSize() != rank || rank > 8) {
        return ge::GRAPH_FAILED;
    }

    yShape->SetDimNum(rank);
    if (*transposeFirst) {
        // permute → reshape：输出 shape = attr shape
        for (size_t i = 0; i < rank; ++i) {
            yShape->SetDim(i, shapeData[i]);
        }
    } else {
        // reshape → permute：输出 shape[i] = attr_shape[perm[i]]
        for (size_t i = 0; i < rank; ++i) {
            const int64_t p = permData[i];
            if (p < 0 || static_cast<size_t>(p) >= rank) {
                return ge::GRAPH_FAILED;
            }
            yShape->SetDim(i, shapeData[p]);
        }
    }
    (void)xShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ConfusionTransposeD).InferShape(InferShapeConfusionTransposeD);

} // namespace ops
