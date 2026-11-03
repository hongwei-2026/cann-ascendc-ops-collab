/*!
 * \file p_relu_grad_update.h
 * \brief Vectorized PReluGradUpdate kernel.
 *
 * P029: 禁止 DtypeBridge 显式特化 / ToFloat / Cast——它们会编进每个 tiling key 的 TU，
 * 导致连 fp32 key 都报 "not support bf16 type cast"（EXP_P024/P027）。
 * - float / half：Compare+Select 原生 dtype，无升降精度
 * - bf16：仅在 if constexpr(bf16) 分支内做位扩展到 fp32 再算（不实例化到其它 key）
 */

#ifndef PRELUGRADUPDATE_H
#define PRELUGRADUPDATE_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "p_relu_grad_update_tiling_data.h"
#include "p_relu_grad_update_tiling_key.h"

namespace NsPReluGradUpdate {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class PReluGradUpdate {
public:
    __aicore__ inline PReluGradUpdate() = default;

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx,
        GM_ADDR update, const PReluGradUpdateTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void BuildWeightVectorNative(int64_t globalStart, int64_t currentNum);
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment) const;
    __aicore__ inline int64_t CmpAlignElems() const;
    __aicore__ inline T BitZero() const;

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradsQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> featuresQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dxQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> updateQueue_;

    // Native float/half path
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> negBuf_;
    TBuf<TPosition::VECCALC> fgBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;

    // bf16 → fp32 scratch（仅 bf16 实例化时 Init）
    TBuf<TPosition::VECCALC> gradsFloatBuf_;
    TBuf<TPosition::VECCALC> featuresFloatBuf_;
    TBuf<TPosition::VECCALC> weightFloatBuf_;
    TBuf<TPosition::VECCALC> dxFloatBuf_;
    TBuf<TPosition::VECCALC> updateFloatBuf_;
    TBuf<TPosition::VECCALC> negFloatBuf_;
    TBuf<TPosition::VECCALC> fgFloatBuf_;

    GlobalTensor<T> gradsGm_;
    GlobalTensor<T> featuresGm_;
    GlobalTensor<T> weightsGm_;
    GlobalTensor<T> dxGm_;
    GlobalTensor<T> updateGm_;

    int64_t totalNum_ = 0;
    int64_t blockStart_ = 0;
    int64_t blockLength_ = 0;
    int64_t tileLen_ = 0;
    int64_t channelNum_ = 0;
    int64_t innerSize_ = 0;
    int64_t weightsPerChannel_ = 0;
    int64_t rowsPerTile_ = 0;
};

template <typename T>
__aicore__ inline int64_t PReluGradUpdate<T>::AlignUp(int64_t value, int64_t alignment) const
{
    return ((value + alignment - 1) / alignment) * alignment;
}

template <typename T>
__aicore__ inline int64_t PReluGradUpdate<T>::CmpAlignElems() const
{
    // Compare/Select：参与元素数 × sizeof(运算类型) 须 256B 对齐。
    // bf16 在 fp32 上算 → 64；float → 64；half 原生 → 128。
    if constexpr (std::is_same<T, half>::value) {
        return 128;
    }
    return 64;
}

template <typename T>
__aicore__ inline T PReluGradUpdate<T>::BitZero() const
{
    T z;
    if constexpr (sizeof(T) == 4) {
        *reinterpret_cast<uint32_t*>(&z) = 0u;
    } else {
        *reinterpret_cast<uint16_t*>(&z) = 0u;
    }
    return z;
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::Init(GM_ADDR grads, GM_ADDR features, GM_ADDR weights,
    GM_ADDR dx, GM_ADDR update, const PReluGradUpdateTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    tileLen_ = tilingData->tileLen;
    channelNum_ = tilingData->channelNum;
    innerSize_ = tilingData->innerSize;
    weightsPerChannel_ = tilingData->weightsPerChannel;
    rowsPerTile_ = tilingData->rowsPerTile;

    blockStart_ = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    blockLength_ = blockStart_ < totalNum_ ? totalNum_ - blockStart_ : 0;
    if (blockLength_ > tilingData->blockFactor) {
        blockLength_ = tilingData->blockFactor;
    }

    gradsGm_.SetGlobalBuffer((__gm__ T*)grads, totalNum_);
    featuresGm_.SetGlobalBuffer((__gm__ T*)features, totalNum_);
    weightsGm_.SetGlobalBuffer((__gm__ T*)weights, weightsPerChannel_ != 0 ? channelNum_ : 1);
    dxGm_.SetGlobalBuffer((__gm__ T*)dx, totalNum_);
    updateGm_.SetGlobalBuffer((__gm__ T*)update, totalNum_);

    const int64_t alignedCount = AlignUp(tileLen_, CmpAlignElems());
    const int64_t alignedMaskBytes = AlignUp(alignedCount, 32);

    pipe_.InitBuffer(gradsQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(featuresQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(dxQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(updateQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(maskBuf_, alignedMaskBytes);

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        pipe_.InitBuffer(gradsFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(featuresFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(weightFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(dxFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(updateFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(negFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(fgFloatBuf_, alignedCount * sizeof(float));
    } else {
        pipe_.InitBuffer(weightBuf_, alignedCount * sizeof(T));
        pipe_.InitBuffer(negBuf_, alignedCount * sizeof(T));
        pipe_.InitBuffer(fgBuf_, alignedCount * sizeof(T));
    }

    (void)rowsPerTile_;
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::BuildWeightVectorNative(int64_t globalStart, int64_t currentNum)
{
    LocalTensor<T> weightLocal = weightBuf_.Get<T>();
    int64_t localOffset = 0;
    while (localOffset < currentNum) {
        const int64_t index = globalStart + localOffset;
        const int64_t channel = (index / innerSize_) % channelNum_;
        const int64_t offsetInChannel = index % innerSize_;
        int64_t segment = innerSize_ - offsetInChannel;
        if (segment > currentNum - localOffset) {
            segment = currentNum - localOffset;
        }
        const T weightValue = weightsGm_.GetValue(channel);
        event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evtSV);
        WaitFlag<HardEvent::S_V>(evtSV);
        Duplicate(weightLocal[localOffset], weightValue, segment);
        localOffset += segment;
    }
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> gradsLocal = gradsQueue_.AllocTensor<T>();
    LocalTensor<T> featuresLocal = featuresQueue_.AllocTensor<T>();

    const int64_t tBlockElements = 32 / static_cast<int64_t>(sizeof(T));
    const int64_t paddedNum = AlignUp(currentNum, tBlockElements);
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0,
        static_cast<uint8_t>(paddedNum - currentNum), BitZero()};
    DataCopyPad(gradsLocal, gradsGm_[blockStart_ + progress], copyParams, padParams);
    DataCopyPad(featuresLocal, featuresGm_[blockStart_ + progress], copyParams, padParams);

    gradsQueue_.EnQue(gradsLocal);
    featuresQueue_.EnQue(featuresLocal);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::Compute(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> gradsLocal = gradsQueue_.DeQue<T>();
    LocalTensor<T> featuresLocal = featuresQueue_.DeQue<T>();
    LocalTensor<T> dxLocal = dxQueue_.AllocTensor<T>();
    LocalTensor<T> updateLocal = updateQueue_.AllocTensor<T>();

    const int64_t cmpCount = AlignUp(currentNum, CmpAlignElems());
    LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        LocalTensor<float> gradsFloat = gradsFloatBuf_.Get<float>();
        LocalTensor<float> featuresFloat = featuresFloatBuf_.Get<float>();
        LocalTensor<float> dxFloat = dxFloatBuf_.Get<float>();
        LocalTensor<float> updateFloat = updateFloatBuf_.Get<float>();
        LocalTensor<float> negLocal = negFloatBuf_.Get<float>();
        LocalTensor<float> fgLocal = fgFloatBuf_.Get<float>();

        for (int64_t i = 0; i < currentNum; ++i) {
            T g = gradsLocal.GetValue(i);
            T f = featuresLocal.GetValue(i);
            uint16_t gb = *reinterpret_cast<uint16_t*>(&g);
            uint16_t fb = *reinterpret_cast<uint16_t*>(&f);
            float gf = 0.0f;
            float ff = 0.0f;
            *reinterpret_cast<uint32_t*>(&gf) = static_cast<uint32_t>(gb) << 16;
            *reinterpret_cast<uint32_t*>(&ff) = static_cast<uint32_t>(fb) << 16;
            gradsFloat.SetValue(i, gf);
            featuresFloat.SetValue(i, ff);
        }
        PipeBarrier<PIPE_V>();

        if (weightsPerChannel_ != 0) {
            LocalTensor<float> weightLocal = weightFloatBuf_.Get<float>();
            int64_t localOffset = 0;
            const int64_t globalStart = blockStart_ + progress;
            while (localOffset < currentNum) {
                const int64_t index = globalStart + localOffset;
                const int64_t channel = (index / innerSize_) % channelNum_;
                const int64_t offsetInChannel = index % innerSize_;
                int64_t segment = innerSize_ - offsetInChannel;
                if (segment > currentNum - localOffset) {
                    segment = currentNum - localOffset;
                }
                T raw = weightsGm_.GetValue(channel);
                uint16_t bits = *reinterpret_cast<uint16_t*>(&raw);
                float weightValue = 0.0f;
                *reinterpret_cast<uint32_t*>(&weightValue) = static_cast<uint32_t>(bits) << 16;
                event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(evtSV);
                WaitFlag<HardEvent::S_V>(evtSV);
                Duplicate(weightLocal[localOffset], weightValue, segment);
                localOffset += segment;
            }
            PipeBarrier<PIPE_V>();
            Mul(negLocal, gradsFloat, weightLocal, currentNum);
        } else {
            T raw = weightsGm_.GetValue(0);
            uint16_t bits = *reinterpret_cast<uint16_t*>(&raw);
            float weightValue = 0.0f;
            *reinterpret_cast<uint32_t*>(&weightValue) = static_cast<uint32_t>(bits) << 16;
            event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evtSV);
            WaitFlag<HardEvent::S_V>(evtSV);
            Muls(negLocal, gradsFloat, weightValue, currentNum);
        }

        Mul(fgLocal, featuresFloat, gradsFloat, currentNum);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> zeroLocal = dxFloat;
        Duplicate(zeroLocal, 0.0f, static_cast<int32_t>(cmpCount));
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, featuresFloat, zeroLocal, CMPMODE::GT, static_cast<uint32_t>(cmpCount));
        PipeBarrier<PIPE_V>();

        Select(dxFloat, maskLocal, gradsFloat, negLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
            static_cast<uint32_t>(cmpCount));
        PipeBarrier<PIPE_V>();

        Duplicate(featuresFloat, 0.0f, static_cast<int32_t>(cmpCount));
        PipeBarrier<PIPE_V>();
        Select(updateFloat, maskLocal, featuresFloat, fgLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
            static_cast<uint32_t>(cmpCount));
        PipeBarrier<PIPE_V>();

        for (int64_t i = 0; i < currentNum; ++i) {
            float dxv = dxFloat.GetValue(i);
            float upv = updateFloat.GetValue(i);
            uint32_t dxu = *reinterpret_cast<uint32_t*>(&dxv);
            uint32_t upu = *reinterpret_cast<uint32_t*>(&upv);
            uint32_t dxLsb = (dxu >> 16) & 1U;
            uint32_t upLsb = (upu >> 16) & 1U;
            uint16_t dxBits = static_cast<uint16_t>((dxu + 0x7FFFU + dxLsb) >> 16);
            uint16_t upBits = static_cast<uint16_t>((upu + 0x7FFFU + upLsb) >> 16);
            T dxOut;
            T upOut;
            *reinterpret_cast<uint16_t*>(&dxOut) = dxBits;
            *reinterpret_cast<uint16_t*>(&upOut) = upBits;
            dxLocal.SetValue(i, dxOut);
            updateLocal.SetValue(i, upOut);
        }
        PipeBarrier<PIPE_V>();
    } else {
        // float / half：原生 dtype，无 Cast / ToFloat
        LocalTensor<T> negLocal = negBuf_.Get<T>();
        LocalTensor<T> fgLocal = fgBuf_.Get<T>();

        if (weightsPerChannel_ != 0) {
            BuildWeightVectorNative(blockStart_ + progress, currentNum);
            PipeBarrier<PIPE_V>();
            LocalTensor<T> weightLocal = weightBuf_.Get<T>();
            Mul(negLocal, gradsLocal, weightLocal, currentNum);
        } else {
            const T weightValue = weightsGm_.GetValue(0);
            event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evtSV);
            WaitFlag<HardEvent::S_V>(evtSV);
            Muls(negLocal, gradsLocal, weightValue, currentNum);
        }

        Mul(fgLocal, featuresLocal, gradsLocal, currentNum);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> zeroLocal = dxLocal;
        Duplicate(zeroLocal, BitZero(), static_cast<int32_t>(cmpCount));
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, featuresLocal, zeroLocal, CMPMODE::GT, static_cast<uint32_t>(cmpCount));
        PipeBarrier<PIPE_V>();

        Select(dxLocal, maskLocal, gradsLocal, negLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
            static_cast<uint32_t>(cmpCount));
        PipeBarrier<PIPE_V>();

        // featuresLocal 已完成 Mul/Compare，复用为 update 正支零向量
        Duplicate(featuresLocal, BitZero(), static_cast<int32_t>(cmpCount));
        PipeBarrier<PIPE_V>();
        Select(updateLocal, maskLocal, featuresLocal, fgLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
            static_cast<uint32_t>(cmpCount));
        PipeBarrier<PIPE_V>();
    }

    dxQueue_.EnQue(dxLocal);
    updateQueue_.EnQue(updateLocal);
    gradsQueue_.FreeTensor(gradsLocal);
    featuresQueue_.FreeTensor(featuresLocal);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> dxLocal = dxQueue_.DeQue<T>();
    LocalTensor<T> updateLocal = updateQueue_.DeQue<T>();
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(dxGm_[blockStart_ + progress], dxLocal, copyParams);
    DataCopyPad(updateGm_[blockStart_ + progress], updateLocal, copyParams);
    dxQueue_.FreeTensor(dxLocal);
    updateQueue_.FreeTensor(updateLocal);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::Process()
{
    if (blockLength_ <= 0 || tileLen_ <= 0) {
        return;
    }
    for (int64_t progress = 0; progress < blockLength_; progress += tileLen_) {
        int64_t currentNum = blockLength_ - progress;
        if (currentNum > tileLen_) {
            currentNum = tileLen_;
        }
        CopyIn(progress, currentNum);
        Compute(progress, currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsPReluGradUpdate
#endif
