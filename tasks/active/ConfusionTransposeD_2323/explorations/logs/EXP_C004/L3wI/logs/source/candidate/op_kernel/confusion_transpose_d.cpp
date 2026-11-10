/*!
 * \file confusion_transpose_d.cpp
 * \brief ConfusionTransposeD 算子 kernel 入口
 */

#include "confusion_transpose_d.h"

enum class ConfusionTransposeDTilingKey : uint32_t
{
    TILING_KEY_CONFUSIONTRANSPOSED_MODE_0 = 0, // 2B
    TILING_KEY_CONFUSIONTRANSPOSED_MODE_1 = 1, // 4B
    TILING_KEY_CONFUSIONTRANSPOSED_MODE_2 = 2, // 1B
    TILING_KEY_CONFUSIONTRANSPOSED_MODE_3 = 3, // 8B
};

template <uint32_t schMode>
__global__ __aicore__ void confusion_transpose_d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ConfusionTransposeDTilingData);
    GET_TILING_DATA_WITH_STRUCT(ConfusionTransposeDTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ConfusionTransposeDTilingKey::TILING_KEY_CONFUSIONTRANSPOSED_MODE_0)) {
        NsConfusionTransposeD::ConfusionTransposeD<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(ConfusionTransposeDTilingKey::TILING_KEY_CONFUSIONTRANSPOSED_MODE_1)) {
        NsConfusionTransposeD::ConfusionTransposeD<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(ConfusionTransposeDTilingKey::TILING_KEY_CONFUSIONTRANSPOSED_MODE_2)) {
        NsConfusionTransposeD::ConfusionTransposeD<int8_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(ConfusionTransposeDTilingKey::TILING_KEY_CONFUSIONTRANSPOSED_MODE_3)) {
        NsConfusionTransposeD::ConfusionTransposeD<int64_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
