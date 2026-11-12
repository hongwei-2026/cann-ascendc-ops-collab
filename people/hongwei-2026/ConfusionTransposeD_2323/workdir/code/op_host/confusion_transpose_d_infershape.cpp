/*!
 * \file confusion_transpose_d_infershape.cpp
 * \brief ConfusionTransposeD 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "confusion_transpose_d_probe.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeConfusionTransposeD(gert::InferShapeContext* context)
{
    ctd_probe::Event probe("infer");
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    probe.Capture(xShape, attrs, -1);
    probe.Stage("missing_input_or_output_shape");
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    probe.Stage("missing_attributes");
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* perm = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const auto* shapeAttr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const bool* transposeFirst = attrs->GetAttrPointer<bool>(2);
    probe.Stage("missing_attribute_value");
    if (perm == nullptr || shapeAttr == nullptr || transposeFirst == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t* permData = reinterpret_cast<const int64_t*>(perm->GetData());
    const int64_t* shapeData = reinterpret_cast<const int64_t*>(shapeAttr->GetData());
    const size_t rank = shapeAttr->GetSize();
    probe.Stage("perm_shape_rank_mismatch");
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
            probe.Stage("perm_out_of_range");
            if (p < 0 || static_cast<size_t>(p) >= rank) {
                return ge::GRAPH_FAILED;
            }
            yShape->SetDim(i, shapeData[p]);
        }
    }
    (void)xShape;
    probe.Output(yShape);
    probe.Success();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ConfusionTransposeD).InferShape(InferShapeConfusionTransposeD);

} // namespace ops
