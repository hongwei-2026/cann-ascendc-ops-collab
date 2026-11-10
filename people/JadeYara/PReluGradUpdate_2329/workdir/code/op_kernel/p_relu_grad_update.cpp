#include "p_relu_grad_update_full_pipeline.h"
#include "p_relu_grad_update_general.h"
#include "p_relu_grad_update_gather.h"
/*!
 * \file p_relu_grad_update.cpp
 * \brief PReluGradUpdate kernel entry.
 */

#include "p_relu_grad_update.h"
#include "p_relu_grad_update_s1.h"
#include "p_relu_grad_update_merged.h"
#include "p_relu_grad_update_resident.h"
#include "p_relu_grad_update_splitc.h"

enum class PReluGradUpdateTilingKey : uint32_t {
    FLOAT = PRELUGRADUPDATE_TPL_FLOAT,
    FLOAT16 = PRELUGRADUPDATE_TPL_FLOAT16,
    BFLOAT16 = PRELUGRADUPDATE_TPL_BF16,
};

template <uint32_t schMode>
__global__ __aicore__ void p_relu_grad_update(GM_ADDR grads, GM_ADDR features, GM_ADDR weights,
    GM_ADDR dx, GM_ADDR update, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (schMode == PRELUGRADUPDATE_TPL_FLOAT16_S1 ||
                  schMode == PRELUGRADUPDATE_TPL_BF16_S1) {
        AscendC::ICachePreLoad(4);
    }
    (void)workspace;
    REGISTER_TILING_DEFAULT(PReluGradUpdateTilingData);
#if PRELU_ENABLE_BF16 && PRELU_ENABLE_FP16
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 15 && TILING_KEY_VAR <= 17", PReluGradUpdateGatherTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 18 && TILING_KEY_VAR <= 20", PReluGradUpdateGeneralTilingData);
#elif PRELU_ENABLE_FP16
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 10 && TILING_KEY_VAR <= 11", PReluGradUpdateGatherTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 12 && TILING_KEY_VAR <= 13", PReluGradUpdateGeneralTilingData);
#else
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 5", PReluGradUpdateGatherTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 6", PReluGradUpdateGeneralTilingData);
#endif

    if constexpr (schMode>=28 && schMode<=30) {
        GET_TILING_DATA_WITH_STRUCT(PReluGradUpdateGeneralTilingData, generalData, tiling);
        if constexpr ((schMode&3U)==0 && std::is_same<DTYPE_GRADS,float>::value) {
            AscendC::TPipe pipe; JadeYaraGeneral::PReluGradUpdate<float> op;
            op.Init(grads,features,weights,dx,update,&generalData,&pipe);op.Process();
        } else if constexpr ((schMode&3U)==1 && std::is_same<DTYPE_GRADS,half>::value) {
            AscendC::TPipe pipe; JadeYaraGeneral::PReluGradUpdate<half> op;
            op.Init(grads,features,weights,dx,update,&generalData,&pipe);op.Process();
        } else if constexpr ((schMode&3U)==2 && std::is_same<DTYPE_GRADS,bfloat16_t>::value) {
            AscendC::TPipe pipe; JadeYaraGeneral::PReluGradUpdate<bfloat16_t> op;
            op.Init(grads,features,weights,dx,update,&generalData,&pipe);op.Process();
        }
    } else if constexpr (schMode>=24 && schMode<=26) {
        GET_TILING_DATA_WITH_STRUCT(PReluGradUpdateGatherTilingData, gatherData, tiling);
        if constexpr ((schMode&3U)==0 && std::is_same<DTYPE_GRADS,float>::value) {
            AscendC::TPipe pipe;JadeYaraContiguous::PReluGradUpdate<float> op;
            op.Init(grads,features,weights,dx,update,&gatherData,&pipe);op.Process();
        } else if constexpr ((schMode&3U)==1 && std::is_same<DTYPE_GRADS,half>::value) {
            AscendC::TPipe pipe;JadeYaraContiguous::PReluGradUpdate<half> op;
            op.Init(grads,features,weights,dx,update,&gatherData,&pipe);op.Process();
        } else if constexpr ((schMode&3U)==2 && std::is_same<DTYPE_GRADS,bfloat16_t>::value) {
            AscendC::TPipe pipe;JadeYaraContiguous::PReluGradUpdate<bfloat16_t> op;
            op.Init(grads,features,weights,dx,update,&gatherData,&pipe);op.Process();
        }
    } else {
    GET_TILING_DATA_WITH_STRUCT(PReluGradUpdateTilingData, tilingData, tiling);



    if constexpr ((schMode & 3U) == static_cast<uint32_t>(PReluGradUpdateTilingKey::FLOAT) &&
                  std::is_same<DTYPE_GRADS, float>::value) {
        // J071: fixed4096 input/output two-slot scheduling, true-TBE Compute unchanged.
        if constexpr (schMode == PRELUGRADUPDATE_TPL_FLOAT_SPLITC) {
            if (tilingData.tileLen == 4096) {
                NsPReluGradUpdateFullPipeline::PReluGradUpdate<float> pipeline;
                pipeline.Init(grads, features, weights, dx, update, &tilingData);
                pipeline.Process();
                return;
            }
        }
        using Kernel = typename std::conditional<(schMode >= 16),
            NsPReluGradUpdateSplitC::PReluGradUpdate<float>,
            typename std::conditional<(schMode >= 12),
            NsPReluGradUpdateResident::PReluGradUpdate<float>,
            typename std::conditional<(schMode >= 8),
                NsPReluGradUpdateMerged::PReluGradUpdate<float>,
                typename std::conditional<(schMode >= 4),
                    NsPReluGradUpdateS1::PReluGradUpdate<float>,
                    NsPReluGradUpdate::PReluGradUpdate<float>>::type>::type>::type>::type;
        Kernel op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
#if PRELU_ENABLE_FP16
    } else if constexpr ((schMode & 3U) == static_cast<uint32_t>(PReluGradUpdateTilingKey::FLOAT16) &&
                  std::is_same<DTYPE_GRADS, half>::value) {
        using Kernel = typename std::conditional<(schMode >= 16),
            NsPReluGradUpdateSplitC::PReluGradUpdate<half>,
            typename std::conditional<(schMode >= 12),
            NsPReluGradUpdateResident::PReluGradUpdate<half>,
            typename std::conditional<(schMode >= 8),
                NsPReluGradUpdateMerged::PReluGradUpdate<half>,
                typename std::conditional<(schMode >= 4),
                    NsPReluGradUpdateS1::PReluGradUpdate<half>,
                    NsPReluGradUpdate::PReluGradUpdate<half>>::type>::type>::type>::type;
        Kernel op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
#endif
#if PRELU_ENABLE_BF16
    } else if constexpr ((schMode & 3U) == static_cast<uint32_t>(PReluGradUpdateTilingKey::BFLOAT16) &&
                  std::is_same<DTYPE_GRADS, bfloat16_t>::value) {
        using Kernel = typename std::conditional<(schMode >= 16),
            NsPReluGradUpdateSplitC::PReluGradUpdate<bfloat16_t>,
            typename std::conditional<(schMode >= 12),
            NsPReluGradUpdateResident::PReluGradUpdate<bfloat16_t>,
            typename std::conditional<(schMode >= 8),
                NsPReluGradUpdateMerged::PReluGradUpdate<bfloat16_t>,
                typename std::conditional<(schMode >= 4),
                    NsPReluGradUpdateS1::PReluGradUpdate<bfloat16_t>,
                    NsPReluGradUpdate::PReluGradUpdate<bfloat16_t>>::type>::type>::type>::type;
        Kernel op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
#endif
    }
    }
}
