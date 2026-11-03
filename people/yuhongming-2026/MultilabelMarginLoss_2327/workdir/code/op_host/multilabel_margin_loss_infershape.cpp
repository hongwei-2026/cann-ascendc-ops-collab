/*!
 * \file multilabel_margin_loss_infershape.cpp
 * \brief MultilabelMarginLoss 算子形状推导实现
 */

#include <cstring>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {
namespace {

constexpr int64_t REDUCTION_NONE = 0;
constexpr int64_t REDUCTION_MEAN = 1;
constexpr int64_t REDUCTION_SUM = 2;
constexpr int64_t REDUCTION_INVALID = -1;

int64_t ParseReduction(const gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return REDUCTION_MEAN;
    }
    const char* reduction = attrs->GetAttrPointer<char>(0);
    if (reduction == nullptr) {
        return REDUCTION_MEAN;
    }
    if (std::strcmp(reduction, "none") == 0) {
        return REDUCTION_NONE;
    }
    if (std::strcmp(reduction, "mean") == 0) {
        return REDUCTION_MEAN;
    }
    if (std::strcmp(reduction, "sum") == 0) {
        return REDUCTION_SUM;
    }
    return REDUCTION_INVALID;
}

} // namespace

static ge::graphStatus InferShapeMultilabelMarginLoss(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* targetShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    gert::Shape* isTargetShape = context->GetOutputShape(1);
    if (xShape == nullptr || targetShape == nullptr || yShape == nullptr || isTargetShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t reduction = ParseReduction(context);
    if (reduction == REDUCTION_INVALID) {
        return ge::GRAPH_FAILED;
    }

    const size_t xRank = xShape->GetDimNum();
    const size_t tRank = targetShape->GetDimNum();
    if (xRank == 0 || xRank > 2 || tRank == 0 || tRank > 2) {
        return ge::GRAPH_FAILED;
    }

    int64_t batch = 1;
    int64_t classNum = 0;
    if (xRank == 1) {
        classNum = xShape->GetDim(0);
    } else {
        batch = xShape->GetDim(0);
        classNum = xShape->GetDim(1);
    }
    if (batch < 0 || classNum < 0) {
        return ge::GRAPH_FAILED;
    }

    // is_target: [N, C]（与 x 对齐；1D 输入时为 [C]）
    if (xRank == 1) {
        isTargetShape->SetDimNum(1);
        isTargetShape->SetDim(0, classNum);
    } else {
        isTargetShape->SetDimNum(2);
        isTargetShape->SetDim(0, batch);
        isTargetShape->SetDim(1, classNum);
    }

    // y: none + 2D → [N]；其余场景输出长度为 1 的向量（判题机不接受 0 维标量）
    if (reduction == REDUCTION_NONE && xRank >= 2) {
        yShape->SetDimNum(1);
        yShape->SetDim(0, batch);
    } else {
        yShape->SetDimNum(1);
        yShape->SetDim(0, 1);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MultilabelMarginLoss).InferShape(InferShapeMultilabelMarginLoss);

} // namespace ops
