/*!
 * \file layer_norm_v3.h
 * \brief LayerNormV3 算子 kernel 类定义
 */

#ifndef LAYERNORMV3_H
#define LAYERNORMV3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "layer_norm_v3_tiling_data.h"
#include "layer_norm_v3_tiling_key.h"

namespace NsLayerNormV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class LayerNormV3 {
public:
    __aicore__ inline LayerNormV3(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, const LayerNormV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void LayerNormV3<T>::Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, const LayerNormV3TilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
}

template <typename T>
__aicore__ inline void LayerNormV3<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
}

template <typename T>
__aicore__ inline void LayerNormV3<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
}

template <typename T>
__aicore__ inline void LayerNormV3<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
}

template <typename T>
__aicore__ inline void LayerNormV3<T>::Process()
{
    // TODO: 实现 Process 逻辑
}

} // namespace NsLayerNormV3
#endif // LAYERNORMV3_H
