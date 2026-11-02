/*!
 * \file p_relu_grad_update.h
 * \brief Vectorized PReluGradUpdate kernel.
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
// Compare / Select 要求参与计算的元素数按 256Byte 对齐。全部中间计算都在 float 上做，
// 故对齐粒度固定为 256/sizeof(float) = 64 个元素。
constexpr int64_t CMP_ALIGN_ELEMS = 256 / static_cast<int64_t>(sizeof(float));

// EXP002/P020: 910B 不支持 bf16 Cast，位扩展/RNE 截断
__aicore__ inline float Bf16ToFloatBits(bfloat16_t v)
{
    uint16_t bits = *reinterpret_cast<uint16_t*>(&v);
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f = 0.0f;
    *reinterpret_cast<uint32_t*>(&f) = u;
    return f;
}

__aicore__ inline bfloat16_t FloatToBf16Rne(float f)
{
    uint32_t u = *reinterpret_cast<uint32_t*>(&f);
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t roundingBias = 0x7FFFU + lsb;
    uint16_t bits = static_cast<uint16_t>((u + roundingBias) >> 16);
    bfloat16_t out;
    *reinterpret_cast<uint16_t*>(&out) = bits;
    return out;
}

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
    __aicore__ inline void BuildWeightVector(int64_t globalStart, int64_t currentNum);
    __aicore__ inline float ReadWeight(int64_t index) const;
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment) const;

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradsQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> featuresQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dxQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> updateQueue_;

    TBuf<TPosition::VECCALC> gradsFloatBuf_;
    TBuf<TPosition::VECCALC> featuresFloatBuf_;
    TBuf<TPosition::VECCALC> weightFloatBuf_;
    TBuf<TPosition::VECCALC> dxFloatBuf_;
    TBuf<TPosition::VECCALC> updateFloatBuf_;
    TBuf<TPosition::VECCALC> negFloatBuf_;    // w*g，Select 的 src1，必须与 dst 不同
    TBuf<TPosition::VECCALC> fgFloatBuf_;     // f*g，Select 的 src1，必须与 dst 不同
    TBuf<TPosition::VECCALC> maskBuf_;

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
__aicore__ inline float PReluGradUpdate<T>::ReadWeight(int64_t index) const
{
    const T value = weightsGm_.GetValue(index);
    if constexpr (std::is_same<T, float>::value) {
        return value;
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        return Bf16ToFloatBits(value);
    } else {
        return ToFloat(value);
    }
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

    // 所有参与 Compare/Select 的 buffer 都按 256Byte（=64 个 float）对齐，
    // 这样尾块把 count 向上取整到 cmpCount 时不会越界。
    const int64_t alignedCount = AlignUp(tileLen_, CMP_ALIGN_ELEMS);
    const int64_t alignedMaskBytes = AlignUp(alignedCount, 32);

    pipe_.InitBuffer(gradsQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(featuresQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(dxQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(updateQueue_, BUFFER_NUM, alignedCount * sizeof(T));
    pipe_.InitBuffer(weightFloatBuf_, alignedCount * sizeof(float));
    pipe_.InitBuffer(negFloatBuf_, alignedCount * sizeof(float));
    pipe_.InitBuffer(fgFloatBuf_, alignedCount * sizeof(float));
    pipe_.InitBuffer(maskBuf_, alignedMaskBytes);

    if constexpr (!std::is_same<T, float>::value) {
        pipe_.InitBuffer(gradsFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(featuresFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(dxFloatBuf_, alignedCount * sizeof(float));
        pipe_.InitBuffer(updateFloatBuf_, alignedCount * sizeof(float));
    }

    if (rowsPerTile_ > 0 && weightsPerChannel_ != 0 && blockLength_ > 0) {
        BuildWeightVector(0, tileLen_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::BuildWeightVector(int64_t globalStart, int64_t currentNum)
{
    LocalTensor<float> weightLocal = weightFloatBuf_.Get<float>();
    int64_t localOffset = 0;
    while (localOffset < currentNum) {
        const int64_t index = globalStart + localOffset;
        const int64_t channel = (index / innerSize_) % channelNum_;
        const int64_t offsetInChannel = index % innerSize_;
        int64_t segment = innerSize_ - offsetInChannel;
        if (segment > currentNum - localOffset) {
            segment = currentNum - localOffset;
        }
        const float weightValue = ReadWeight(channel);
        // CANN 9.1: SyncFunc 不存在；标量→向量用 SetFlag/WaitFlag
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

    // isPad=true 把 32B 块内的尾部补 0，避免尾块 lane 上残留 UB 垃圾。
    const int64_t tBlockElements = 32 / static_cast<int64_t>(sizeof(T));
    const int64_t paddedNum = AlignUp(currentNum, tBlockElements);
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0,
        static_cast<uint8_t>(paddedNum - currentNum), static_cast<T>(0)};
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
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            for (int64_t i = 0; i < currentNum; ++i) {
                gradsFloat.SetValue(i, Bf16ToFloatBits(gradsLocal.GetValue(i)));
                featuresFloat.SetValue(i, Bf16ToFloatBits(featuresLocal.GetValue(i)));
            }
        } else {
            Cast(gradsFloat, gradsLocal, RoundMode::CAST_NONE, currentNum);
            Cast(featuresFloat, featuresLocal, RoundMode::CAST_NONE, currentNum);
        }
        PipeBarrier<PIPE_V>();
    }

    // Compare/Select 的 count 必须按 256Byte 对齐；buffer 已按 alignedCount 申请，
    // 多算出来的尾 lane 不会被 CopyOut 写回。
    const int64_t cmpCount = AlignUp(currentNum, CMP_ALIGN_ELEMS);

    // 负分支候选 w*g 与 f*g 必须写到独立 buffer：Select 不保证逐 lane 原子，
    // 若 dst 与 src1 同址，src1 会在写 dst 的过程中被 src0 覆盖，
    // 导致 mask=0 的 lane 读到 src0（实测现象：负分支全错、正分支全对）。
    LocalTensor<float> negLocal = negFloatBuf_.Get<float>();
    LocalTensor<float> fgLocal = fgFloatBuf_.Get<float>();

    if (weightsPerChannel_ != 0) {
        if (rowsPerTile_ == 0) {
            BuildWeightVector(blockStart_ + progress, currentNum);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float> weightLocal = weightFloatBuf_.Get<float>();
        Mul(negLocal, gradsFloat, weightLocal, currentNum);
    } else {
        const float weightValue = ReadWeight(0);
        event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evtSV);
        WaitFlag<HardEvent::S_V>(evtSV);
        Muls(negLocal, gradsFloat, weightValue, currentNum);
    }

    Mul(fgLocal, featuresFloat, gradsFloat, currentNum);
    LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
    // 必须用浮点 GT 比较：NaN 在任何比较下为假，会落入负分支，与 golden 的 np.where 一致；
    // 改用 LE 取反会把 NaN 也判成假，两个分支都错。
    CompareScalar(maskLocal, featuresFloat, 0.0f, CMPMODE::GT, static_cast<uint32_t>(cmpCount));
    PipeBarrier<PIPE_V>();

    // selMask=1 取 src0：正分支取 grads，非正分支取 w*grads。
    Select(dxFloat, maskLocal, gradsFloat, negLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
        static_cast<uint32_t>(cmpCount));

    // featuresFloat 在 Mul/Compare 之后已无用，复用为全 0 向量（它不是任何 Select 的 dst）。
    LocalTensor<float> zeroLocal = featuresFloat;
    Duplicate(zeroLocal, 0.0f, static_cast<int32_t>(cmpCount));
    PipeBarrier<PIPE_V>();
    Select(updateFloat, maskLocal, zeroLocal, fgLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
        static_cast<uint32_t>(cmpCount));
    PipeBarrier<PIPE_V>();

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        for (int64_t i = 0; i < currentNum; ++i) {
            dxLocal.SetValue(i, FloatToBf16Rne(dxFloat.GetValue(i)));
            updateLocal.SetValue(i, FloatToBf16Rne(updateFloat.GetValue(i)));
        }
        PipeBarrier<PIPE_V>();
    } else if constexpr (!std::is_same<T, float>::value) {
        Cast(dxLocal, dxFloat, RoundMode::CAST_RINT, currentNum);
        Cast(updateLocal, updateFloat, RoundMode::CAST_RINT, currentNum);
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
