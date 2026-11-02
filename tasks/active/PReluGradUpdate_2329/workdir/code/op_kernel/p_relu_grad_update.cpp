/*!
 * \file p_relu_grad_update.cpp
 * \brief Kernel entry — schMode 0=fp32, 1=fp16, 2=bf16
 */

#include "p_relu_grad_update.h"

enum class PReluGradUpdateTilingKey : uint32_t {
    TILING_KEY_FLOAT = 0,
    TILING_KEY_HALF = 1,
    TILING_KEY_BF16 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void p_relu_grad_update(
    GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx, GM_ADDR update, GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PReluGradUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(PReluGradUpdateTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(PReluGradUpdateTilingKey::TILING_KEY_FLOAT)) {
        NsPReluGradUpdate::PReluGradUpdate<float> op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(PReluGradUpdateTilingKey::TILING_KEY_HALF)) {
        NsPReluGradUpdate::PReluGradUpdate<half> op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(PReluGradUpdateTilingKey::TILING_KEY_BF16)) {
        NsPReluGradUpdate::PReluGradUpdate<bfloat16_t> op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
    }
    (void)workspace;
}
