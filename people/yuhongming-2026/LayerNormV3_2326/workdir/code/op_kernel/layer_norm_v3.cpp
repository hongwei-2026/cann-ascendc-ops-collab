/*!
 * \file layer_norm_v3.cpp
 * \brief LayerNormV3 kernel 入口。schMode 选择 x 与 gamma 的类型，归约在 float32 完成。
 */

#include "layer_norm_v3.h"

enum class LayerNormV3TilingKey : uint32_t
{
    TILING_KEY_LAYERNORMV3_MODE_0 = 0, // x/gamma = fp16
    TILING_KEY_LAYERNORMV3_MODE_1 = 1, // x/gamma = fp32
    TILING_KEY_LAYERNORMV3_MODE_2 = 2, // x/gamma = bf16
    TILING_KEY_LAYERNORMV3_MODE_3 = 3, // x = fp16, gamma/mean/rstd = fp32
    TILING_KEY_LAYERNORMV3_MODE_4 = 4, // x = bf16, gamma/mean/rstd = fp32
};

template <uint32_t schMode>
__global__ __aicore__ void layer_norm_v3(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LayerNormV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(LayerNormV3TilingKey::TILING_KEY_LAYERNORMV3_MODE_0)) {
        NsLayerNormV3::LayerNormV3<half, half> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(LayerNormV3TilingKey::TILING_KEY_LAYERNORMV3_MODE_1)) {
        NsLayerNormV3::LayerNormV3<float, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(LayerNormV3TilingKey::TILING_KEY_LAYERNORMV3_MODE_2)) {
        NsLayerNormV3::LayerNormV3<bfloat16_t, bfloat16_t> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(LayerNormV3TilingKey::TILING_KEY_LAYERNORMV3_MODE_3)) {
        NsLayerNormV3::LayerNormV3<half, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(LayerNormV3TilingKey::TILING_KEY_LAYERNORMV3_MODE_4)) {
        NsLayerNormV3::LayerNormV3<bfloat16_t, float> op;
        op.Init(x, gamma, beta, y, mean, rstd, &tilingData);
        op.Process();
    }
    (void)workspace;
}
