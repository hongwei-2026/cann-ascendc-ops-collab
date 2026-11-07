/*!
 * \file pad_v3_grad.cpp
 * \brief PadV3Grad 算子 kernel 入口
 */

#include "pad_v3_grad.h"

enum class PadV3GradTilingKey : uint32_t
{
    TILING_KEY_PADV3GRAD_MODE_0 = 0,
    TILING_KEY_PADV3GRAD_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void pad_v3_grad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PadV3GradTilingData);
    GET_TILING_DATA_WITH_STRUCT(PadV3GradTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(PadV3GradTilingKey::TILING_KEY_PADV3GRAD_MODE_0)) {
        NsPadV3Grad::PadV3Grad<half> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(PadV3GradTilingKey::TILING_KEY_PADV3GRAD_MODE_1)) {
        NsPadV3Grad::PadV3Grad<float> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    }
}
