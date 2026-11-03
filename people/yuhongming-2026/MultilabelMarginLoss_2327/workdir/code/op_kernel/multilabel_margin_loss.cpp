/*!
 * \file multilabel_margin_loss.cpp
 * \brief MultilabelMarginLoss 算子 kernel 入口
 */

#include "multilabel_margin_loss.h"

enum class MultilabelMarginLossTilingKey : uint32_t
{
    TILING_KEY_MULTILABELMARGINLOSS_MODE_0 = 0, // float16
    TILING_KEY_MULTILABELMARGINLOSS_MODE_1 = 1, // float32
    TILING_KEY_MULTILABELMARGINLOSS_MODE_2 = 2, // bfloat16
};

template <uint32_t schMode>
__global__ __aicore__ void multilabel_margin_loss(GM_ADDR x, GM_ADDR target, GM_ADDR y, GM_ADDR is_target,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MultilabelMarginLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(MultilabelMarginLossTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(MultilabelMarginLossTilingKey::TILING_KEY_MULTILABELMARGINLOSS_MODE_0)) {
        NsMultilabelMarginLoss::MultilabelMarginLoss<half> op;
        op.Init(x, target, y, is_target, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(MultilabelMarginLossTilingKey::TILING_KEY_MULTILABELMARGINLOSS_MODE_1)) {
        NsMultilabelMarginLoss::MultilabelMarginLoss<float> op;
        op.Init(x, target, y, is_target, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(MultilabelMarginLossTilingKey::TILING_KEY_MULTILABELMARGINLOSS_MODE_2)) {
        NsMultilabelMarginLoss::MultilabelMarginLoss<bfloat16_t> op;
        op.Init(x, target, y, is_target, workspace, &tilingData);
        op.Process();
    }
}
