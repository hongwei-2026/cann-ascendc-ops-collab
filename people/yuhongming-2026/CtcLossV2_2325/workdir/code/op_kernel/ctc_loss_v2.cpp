/*!
 * \file ctc_loss_v2.cpp
 * \brief CtcLossV2 算子 kernel 入口（910B 样例 dtype=float32）
 */

#include "ctc_loss_v2.h"

enum class CtcLossV2TilingKey : uint32_t
{
    TILING_KEY_CTCLOSSV2_MODE_0 = 0, // 预留（与模板 key 对齐；实现走 float）
    TILING_KEY_CTCLOSSV2_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void ctc_loss_v2(
    GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths,
    GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CtcLossV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(CtcLossV2TilingData, tilingData, tiling);
    // def 仅注册 DT_FLOAT；两种 schMode 均走 float 实现，保证模板 key 可链上
    (void)schMode;
    NsCtcLossV2::CtcLossV2<float> op;
    op.Init(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, &tilingData);
    op.Process();
}
