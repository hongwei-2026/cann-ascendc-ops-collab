/*!
 * \file p_relu_grad_update.cpp
 * \brief PReluGradUpdate kernel entry.
 */

#include "p_relu_grad_update.h"

enum class PReluGradUpdateTilingKey : uint32_t {
    FLOAT = PRELUGRADUPDATE_TPL_FLOAT,
    FLOAT16 = PRELUGRADUPDATE_TPL_FLOAT16,
    BFLOAT16 = PRELUGRADUPDATE_TPL_BF16,
};

template <uint32_t schMode>
__global__ __aicore__ void p_relu_grad_update(GM_ADDR grads, GM_ADDR features, GM_ADDR weights,
    GM_ADDR dx, GM_ADDR update, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(PReluGradUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(PReluGradUpdateTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(PReluGradUpdateTilingKey::FLOAT)) {
        NsPReluGradUpdate::PReluGradUpdate<float> op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(PReluGradUpdateTilingKey::FLOAT16)) {
        NsPReluGradUpdate::PReluGradUpdate<half> op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(PReluGradUpdateTilingKey::BFLOAT16)) {
        NsPReluGradUpdate::PReluGradUpdate<bfloat16_t> op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
    }
}
