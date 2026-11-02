/*!
 * \file p_relu_grad_update.h
 * \brief PReluGradUpdate 算子 kernel 类定义
 */

#ifndef PRELUGRADUPDATE_H
#define PRELUGRADUPDATE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "p_relu_grad_update_tiling_data.h"
#include "p_relu_grad_update_tiling_key.h"

namespace NsPReluGradUpdate {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class PReluGradUpdate {
public:
    __aicore__ inline PReluGradUpdate(){};

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx, GM_ADDR update, const PReluGradUpdateTilingData* tilingData);
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
__aicore__ inline void PReluGradUpdate<T>::Init(GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx, GM_ADDR update, const PReluGradUpdateTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::Process()
{
    // TODO: 实现 Process 逻辑
}

} // namespace NsPReluGradUpdate
#endif // PRELUGRADUPDATE_H
