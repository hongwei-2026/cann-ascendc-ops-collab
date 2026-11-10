/*!
 * \file p_relu_grad_update.h
 * \brief Vectorized PReluGradUpdate kernel.
 *
 * P030: case4 ACL 507035。half 不用原生 Compare/Select，仅在 if constexpr(half)
 * 内 Cast↔fp32；bf16 位转换只写在 if constexpr(bf16) 内，并补 S_V/S_MTE3。
 * 禁止把 half/bf16 辅助函数做成对所有 T 都会实例化的成员（会污染 fp32 key）。
 * P100: zero unused workspace, use full-core 32B tiling, and keep each
 * per-channel tile inside one channel so scalar Muls replaces weight expansion.
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
constexpr int64_t CMP_ALIGN_FLOAT = 64;

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
    __aicore__ inline void SyncSV();
    __aicore__ inline void SyncSMTE3();
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment) const;

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradsQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> featuresQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dxQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> updateQueue_;

    TBuf<TPosition::VECCALC> fgFloatBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> gradsFloatBuf_;
    TBuf<TPosition::VECCALC> featuresFloatBuf_;
    TBuf<TPosition::VECCALC> dxFloatBuf_;
    TBuf<TPosition::VECCALC> updateFloatBuf_;

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
__aicore__ inline void PReluGradUpdate<T>::SyncSV()
{
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(evt);
    WaitFlag<HardEvent::S_V>(evt);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::SyncSMTE3()
{
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evt);
    WaitFlag<HardEvent::S_MTE3>(evt);
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

    const int64_t alignedCount = AlignUp(tileLen_, std::is_same<T, half>::value ? 128 : CMP_ALIGN_FLOAT);
    const int64_t alignedMaskBytes = AlignUp(alignedCount, 32);

    pipe_.InitBuffer(gradsQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(featuresQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(dxQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(updateQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(maskBuf_, alignedMaskBytes);
    pipe_.InitBuffer(fgFloatBuf_, alignedCount * sizeof(float));

    if constexpr (!std::is_same<T, float>::value) {
        pipe_.InitBuffer(gradsFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(featuresFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(dxFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(updateFloatBuf_, alignedCount * sizeof(float));
    }

    (void)rowsPerTile_;
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> gradsLocal = gradsQueue_.AllocTensor<T>();
    LocalTensor<T> featuresLocal = featuresQueue_.AllocTensor<T>();

    const int64_t tBlockElements = 32 / static_cast<int64_t>(sizeof(T));
    const int64_t paddedNum = AlignUp(currentNum, tBlockElements);
    T padVal;
    if constexpr (sizeof(T) == 4) {
        *reinterpret_cast<uint32_t*>(&padVal) = 0u;
    } else {
        *reinterpret_cast<uint16_t*>(&padVal) = 0u;
    }
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0,
        static_cast<uint8_t>(paddedNum - currentNum), padVal};
    DataCopyPad(gradsLocal, gradsGm_[blockStart_ + progress], copyParams, padParams);
    DataCopyPad(featuresLocal, featuresGm_[blockStart_ + progress], copyParams, padParams);

    gradsQueue_.EnQue(gradsLocal);
    featuresQueue_.EnQue(featuresLocal);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::Compute(int64_t progress, int64_t currentNum)
{
    if constexpr (std::is_same<T, half>::value) {

    LocalTensor<T> gradsLocal = gradsQueue_.DeQue<T>();
    LocalTensor<T> featuresLocal = featuresQueue_.DeQue<T>();
    LocalTensor<T> dxLocal = dxQueue_.AllocTensor<T>();
    LocalTensor<T> updateLocal = updateQueue_.AllocTensor<T>();
    const int64_t cmpCount = AlignUp(currentNum, 128);
    LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
    LocalTensor<T> fgLocal = fgFloatBuf_.Get<T>();
    T scalarWeight;
    const int64_t globalStart = blockStart_ + progress;
    const int64_t weightIndex = weightsPerChannel_ != 0 ? (globalStart / innerSize_) % channelNum_ : 0;
    scalarWeight = weightsGm_.GetValue(weightIndex);
    SyncSV();
    Duplicate(dxLocal, static_cast<T>(0.0f), static_cast<int32_t>(cmpCount));
    Duplicate(fgLocal, static_cast<T>(1.0f), static_cast<int32_t>(cmpCount));
    PipeBarrier<PIPE_V>();
    Compare(maskLocal, featuresLocal, dxLocal, CMPMODE::GT, static_cast<uint32_t>(cmpCount));
    PipeBarrier<PIPE_V>();
    Select(updateLocal, maskLocal, dxLocal, fgLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(cmpCount));
    PipeBarrier<PIPE_V>();
    // True TBE chain: negative mask multiplies x/w; do not select away invalid products.
    Mul(featuresLocal, updateLocal, featuresLocal, currentNum);
    Muls(fgLocal, updateLocal, static_cast<T>(-1.0f), currentNum);
    PipeBarrier<PIPE_V>();
    Adds(fgLocal, fgLocal, static_cast<T>(1.0f), currentNum);
    Muls(dxLocal, updateLocal, scalarWeight, currentNum);
    PipeBarrier<PIPE_V>();
    Add(dxLocal, dxLocal, fgLocal, currentNum);
    PipeBarrier<PIPE_V>();
    Mul(dxLocal, dxLocal, gradsLocal, currentNum);
    Mul(updateLocal, featuresLocal, gradsLocal, currentNum);
    PipeBarrier<PIPE_V>();
    // dx is already quantized to half; its Inf/NaN must propagate into update.
    Muls(fgLocal, dxLocal, static_cast<T>(0.0f), currentNum);
    PipeBarrier<PIPE_V>();
    Add(updateLocal, updateLocal, fgLocal, currentNum);
    PipeBarrier<PIPE_V>();
    dxQueue_.EnQue(dxLocal);
    updateQueue_.EnQue(updateLocal);
    gradsQueue_.FreeTensor(gradsLocal);
    featuresQueue_.FreeTensor(featuresLocal);
    } else {

    LocalTensor<T> gradsLocal = gradsQueue_.DeQue<T>();
    LocalTensor<T> featuresLocal = featuresQueue_.DeQue<T>();
    LocalTensor<T> dxLocal = dxQueue_.AllocTensor<T>();
    LocalTensor<T> updateLocal = updateQueue_.AllocTensor<T>();

    const int64_t cmpCount = AlignUp(currentNum, CMP_ALIGN_FLOAT);
    LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
    LocalTensor<float> fgLocal = fgFloatBuf_.Get<float>();

    LocalTensor<float> gradsFloat;
    LocalTensor<float> featuresFloat;
    LocalTensor<float> dxFloat;
    LocalTensor<float> updateFloat;

    if constexpr (std::is_same<T, float>::value) {
        gradsFloat = gradsLocal.template ReinterpretCast<float>();
        featuresFloat = featuresLocal.template ReinterpretCast<float>();
        dxFloat = dxLocal.template ReinterpretCast<float>();
        updateFloat = updateLocal.template ReinterpretCast<float>();
    } else {
        gradsFloat = gradsFloatBuf_.Get<float>();
        featuresFloat = featuresFloatBuf_.Get<float>();
        dxFloat = dxFloatBuf_.Get<float>();
        updateFloat = updateFloatBuf_.Get<float>();

        if constexpr (std::is_same<T, half>::value) {
            Cast(gradsFloat, gradsLocal, RoundMode::CAST_NONE, currentNum);
            Cast(featuresFloat, featuresLocal, RoundMode::CAST_NONE, currentNum);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(gradsFloat, gradsLocal, RoundMode::CAST_NONE, currentNum);
            Cast(featuresFloat, featuresLocal, RoundMode::CAST_NONE, currentNum);
            PipeBarrier<PIPE_V>();
        }
    }

    float scalarWeight = 0.0f;
    const int64_t globalStart = blockStart_ + progress;
    const int64_t weightIndex = weightsPerChannel_ != 0
        ? (globalStart / innerSize_) % channelNum_
        : 0;
    scalarWeight = 0.0f;
    if constexpr (std::is_same<T, float>::value) {
        scalarWeight = weightsGm_.GetValue(weightIndex);
    } else if constexpr (std::is_same<T, half>::value) {
        half h = weightsGm_.GetValue(weightIndex);
        uint16_t hb = *reinterpret_cast<uint16_t*>(&h);
        uint32_t sign = (static_cast<uint32_t>(hb) & 0x8000u) << 16;
        uint32_t exp = (static_cast<uint32_t>(hb) >> 10) & 0x1Fu;
        uint32_t man = static_cast<uint32_t>(hb) & 0x3FFu;
        uint32_t outBits;
        if (exp == 0) {
            if (man == 0) {
                outBits = sign;
            } else {
                exp = 127 - 15 + 1;
                while ((man & 0x400u) == 0) {
                    man <<= 1;
                    --exp;
                }
                man &= 0x3FFu;
                outBits = sign | (exp << 23) | (man << 13);
            }
        } else if (exp == 31) {
            outBits = sign | 0x7F800000u | (man << 13);
        } else {
            outBits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
        }
        *reinterpret_cast<uint32_t*>(&scalarWeight) = outBits;
    } else {
        T raw = weightsGm_.GetValue(weightIndex);
        uint16_t bits = *reinterpret_cast<uint16_t*>(&raw);
        *reinterpret_cast<uint32_t*>(&scalarWeight) = static_cast<uint32_t>(bits) << 16;
    }
    SyncSV();
    Duplicate(dxFloat, 0.0f, static_cast<int32_t>(cmpCount));
    Duplicate(fgLocal, 1.0f, static_cast<int32_t>(cmpCount));
    PipeBarrier<PIPE_V>();
    Compare(maskLocal, featuresFloat, dxFloat, CMPMODE::GT, static_cast<uint32_t>(cmpCount));
    PipeBarrier<PIPE_V>();
    Select(updateFloat, maskLocal, dxFloat, fgLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(cmpCount));
    PipeBarrier<PIPE_V>();
    Mul(featuresFloat, updateFloat, featuresFloat, currentNum);
    Muls(fgLocal, updateFloat, -1.0f, currentNum);
    PipeBarrier<PIPE_V>();
    Adds(fgLocal, fgLocal, 1.0f, currentNum);
    Muls(dxFloat, updateFloat, scalarWeight, currentNum);
    PipeBarrier<PIPE_V>();
    Add(dxFloat, dxFloat, fgLocal, currentNum);
    PipeBarrier<PIPE_V>();
    Mul(dxFloat, dxFloat, gradsFloat, currentNum);
    Mul(updateFloat, featuresFloat, gradsFloat, currentNum);
    PipeBarrier<PIPE_V>();
    if constexpr (!std::is_same<T, float>::value) {
        Cast(dxLocal, dxFloat, RoundMode::CAST_RINT, currentNum);
        PipeBarrier<PIPE_V>();
        Cast(dxFloat, dxLocal, RoundMode::CAST_NONE, currentNum);
        PipeBarrier<PIPE_V>();
    }
    Muls(fgLocal, dxFloat, 0.0f, currentNum);
    PipeBarrier<PIPE_V>();
    Add(updateFloat, updateFloat, fgLocal, currentNum);
    PipeBarrier<PIPE_V>();
    if constexpr (!std::is_same<T, float>::value) {
        Cast(updateLocal, updateFloat, RoundMode::CAST_RINT, currentNum);
        PipeBarrier<PIPE_V>();
    }
    dxQueue_.EnQue(dxLocal);
    updateQueue_.EnQue(updateLocal);
    gradsQueue_.FreeTensor(gradsLocal);
    featuresQueue_.FreeTensor(featuresLocal);
    }
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
    if (blockLength_ <= 0 || tileLen_ <= 0 || innerSize_ <= 0 || channelNum_ <= 0) {
        return;
    }
    // Must advance by currentNum: channel clamping can make currentNum < tileLen_.
    // Using progress += tileLen_ would skip the remainder of the channel (WA).
    for (int64_t progress = 0; progress < blockLength_;) {
        int64_t currentNum = blockLength_ - progress;
        if (currentNum > tileLen_) {
            currentNum = tileLen_;
        }
        if (weightsPerChannel_ != 0) {
            const int64_t globalStart = blockStart_ + progress;
            const int64_t channelRemain = innerSize_ - globalStart % innerSize_;
            if (currentNum > channelRemain) {
                currentNum = channelRemain;
            }
        }
        CopyIn(progress, currentNum);
        Compute(progress, currentNum);
        CopyOut(progress, currentNum);
        progress += currentNum;
    }
}

} // namespace NsPReluGradUpdate
#endif