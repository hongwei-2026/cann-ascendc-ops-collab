/*!
 * \file quant_update_scatter.h
 * \brief QuantUpdateScatter 算子 kernel
 *
 * 语义对齐 samples/reference.py / 内置 AscendC：
 *   y = var.copy()
 *   for p, idx in enumerate(indices):
 *       for k in range(updates.shape[axis]):          # axis 仅支持 -2
 *           y[p, ..., idx+k, :] = cast_int8_rint(
 *               fp32(updates[p, ..., k, :]) / fp32(scales) + fp32(zero_points))
 *   quant_axis 仅支持 -1；除法在 fp32；CAST_RINT；结果 clip 到 [-128, 127]。
 *   reciprocal_scale=true 时改为乘法。
 */

#ifndef QUANTUPDATESCATTER_H
#define QUANTUPDATESCATTER_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_update_scatter_tiling_data.h"
#include "quant_update_scatter_tiling_key.h"

namespace NsQuantUpdateScatter {

using namespace AscendC;

constexpr int32_t VEC_ALIGN_FP32 = 8;
constexpr int32_t DMA_ALIGN = 32;
constexpr int32_t INDEX_CACHE_CAP = 2048;

__aicore__ inline int32_t AlignUpInt(int32_t value, int32_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
class QuantUpdateScatter {
public:
    __aicore__ inline QuantUpdateScatter() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR quant_scales,
        GM_ADDR quant_zero_points, GM_ADDR var_out, const QuantUpdateScatterTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyVarToOut();
    template <typename T>
    __aicore__ inline void LoadTile(LocalTensor<T> local, GlobalTensor<T>& global, int64_t offset, int32_t len);
    __aicore__ inline void LoadIndices();
    __aicore__ inline void LoadQuantParams(int64_t column, int32_t len);
    __aicore__ inline void QuantOneTile(int64_t srcOffset, int64_t dstOffset, int32_t len);
    __aicore__ inline int64_t ReadIndex(int64_t p);

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> updatesBuf_;
    TBuf<TPosition::VECCALC> scalesRawBuf_;
    TBuf<TPosition::VECCALC> zpRawBuf_;
    TBuf<TPosition::VECCALC> scalesBuf_;
    TBuf<TPosition::VECCALC> zpBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> int16Buf_;
    TBuf<TPosition::VECCALC> halfBuf_;
    TBuf<TPosition::VECCALC> outBuf_;

    TBuf<TPosition::VECCALC> indicesBuf_;

    GlobalTensor<int8_t> varGm_;
    GlobalTensor<int8_t> outGm_;
    GlobalTensor<UpdatesT> updatesGm_;
    GlobalTensor<ScalesT> scalesGm_;
    GlobalTensor<OffsetsT> zpGm_;
    GlobalTensor<IndicesT> indicesGm_;

    int64_t coreNum_ = 1;
    int64_t indexCacheLen_ = 0;
    int64_t totalNum_ = 0;
    int64_t batchStart_ = 0;
    int64_t batchLen_ = 0;
    int64_t midSize_ = 1;
    int64_t updateAxis_ = 0;
    int64_t varAxis_ = 0;
    int64_t lastDim_ = 0;
    int64_t indexElements_ = 0;
    int64_t hasZeroPoints_ = 0;
    int64_t reciprocalScale_ = 0;
    int64_t ubFactor_ = 64;
    int32_t bufElems_ = 64;
    uint64_t varAddr_ = 0;
    uint64_t outAddr_ = 0;
};

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::Init(GM_ADDR var, GM_ADDR indices,
    GM_ADDR updates, GM_ADDR quant_scales, GM_ADDR quant_zero_points, GM_ADDR var_out,
    const QuantUpdateScatterTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    coreNum_ = tilingData->coreNum > 0 ? tilingData->coreNum : 1;
    midSize_ = tilingData->midSize > 0 ? tilingData->midSize : 1;
    updateAxis_ = tilingData->updateAxis;
    varAxis_ = tilingData->varAxis;
    lastDim_ = tilingData->lastDim;
    indexElements_ = tilingData->indexElements;
    hasZeroPoints_ = tilingData->hasZeroPoints;
    reciprocalScale_ = tilingData->reciprocalScale;
    ubFactor_ = tilingData->ubFactor > 0 ? tilingData->ubFactor : 64;
    bufElems_ = AlignUpInt(static_cast<int32_t>(ubFactor_), DMA_ALIGN);

    const int64_t blockIdx = GetBlockIdx();
    const int64_t factor = tilingData->blockFactor > 0 ? tilingData->blockFactor : tilingData->batchSize;
    batchStart_ = blockIdx * factor;
    const int64_t remain = tilingData->batchSize - batchStart_;
    batchLen_ = remain > factor ? factor : (remain > 0 ? remain : 0);

    const int64_t indexRemain = indexElements_ - batchStart_;
    indexCacheLen_ = indexRemain < batchLen_ ? (indexRemain > 0 ? indexRemain : 0) : batchLen_;
    if (indexCacheLen_ > INDEX_CACHE_CAP) {
        indexCacheLen_ = 0; // bounded cache; retain scalar GM fallback
    }
    varAddr_ = reinterpret_cast<uint64_t>(var);
    outAddr_ = reinterpret_cast<uint64_t>(var_out);
    varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(var));
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(var_out));
    updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ UpdatesT*>(updates));
    scalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesT*>(quant_scales));
    zpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OffsetsT*>(quant_zero_points));
    indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IndicesT*>(indices));

    const int32_t n = bufElems_;
    pipe_.InitBuffer(updatesBuf_, n * static_cast<int32_t>(sizeof(UpdatesT)));
    pipe_.InitBuffer(scalesRawBuf_, n * 4); // float or bf16 padded
    pipe_.InitBuffer(zpRawBuf_, n * 4);     // int32 or bf16
    pipe_.InitBuffer(scalesBuf_, n * static_cast<int32_t>(sizeof(float)));
    pipe_.InitBuffer(zpBuf_, n * static_cast<int32_t>(sizeof(float)));
    pipe_.InitBuffer(workBuf_, n * static_cast<int32_t>(sizeof(float)));
    pipe_.InitBuffer(int16Buf_, n * static_cast<int32_t>(sizeof(int16_t)));
    pipe_.InitBuffer(halfBuf_, n * static_cast<int32_t>(sizeof(half)));
    pipe_.InitBuffer(outBuf_, n * static_cast<int32_t>(sizeof(int8_t)));
    if (indexCacheLen_ > 0) {
        pipe_.InitBuffer(indicesBuf_, AlignUpInt(static_cast<int32_t>(indexCacheLen_ * sizeof(IndicesT)), DMA_ALIGN));
    }
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
template <typename T>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::LoadTile(
    LocalTensor<T> local, GlobalTensor<T>& global, int64_t offset, int32_t len)
{
    constexpr int32_t blockElems = DMA_ALIGN / sizeof(T);
    const int32_t rem = len % blockElems;
    const uint8_t rightPad = rem == 0 ? 0 : static_cast<uint8_t>(blockElems - rem);
    DataCopyExtParams params = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad = {true, 0, rightPad, T(0)};
    DataCopyPad(local, global[offset], params, pad);
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::CopyVarToOut()
{
    if (totalNum_ <= 0 || varAddr_ == outAddr_) {
        return;
    }
    // Whole 32B blocks per core prevent copy-boundary overlap.
    const int64_t blocks = totalNum_ / DMA_ALIGN + (totalNum_ % DMA_ALIGN != 0);
    const int64_t blocksPerCore = blocks / coreNum_ + (blocks % coreNum_ != 0);
    int64_t offset = static_cast<int64_t>(GetBlockIdx()) * blocksPerCore * DMA_ALIGN;
    int64_t end = offset + blocksPerCore * DMA_ALIGN;
    end = end < totalNum_ ? end : totalNum_;
    LocalTensor<int8_t> tmp = outBuf_.Get<int8_t>();
    while (offset < end) {
        const int64_t remaining = end - offset;
        const int32_t len = remaining > bufElems_ ? bufElems_ : static_cast<int32_t>(remaining);
        LoadTile(tmp, varGm_, offset, len);
        const event_t loaded = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(loaded);
        WaitFlag<HardEvent::MTE2_MTE3>(loaded);
        DataCopyExtParams params = {1, static_cast<uint32_t>(len), 0, 0, 0};
        DataCopyPad(outGm_[offset], tmp, params);
        const event_t stored = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(stored);
        WaitFlag<HardEvent::MTE3_MTE2>(stored);
        offset += len;
    }
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::LoadIndices()
{
    if (indexCacheLen_ > 0) {
        LoadTile(indicesBuf_.Get<IndicesT>(), indicesGm_, batchStart_, static_cast<int32_t>(indexCacheLen_));
        const event_t loaded = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(loaded);
        WaitFlag<HardEvent::MTE2_S>(loaded);
    }
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline int64_t QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::ReadIndex(int64_t p)
{
    if (p < 0 || p >= indexElements_) {
        return 0;
    }
    if (indexCacheLen_ > 0) {
        return static_cast<int64_t>(indicesBuf_.Get<IndicesT>().GetValue(p - batchStart_));
    }
    return static_cast<int64_t>(indicesGm_.GetValue(p));
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::LoadQuantParams(
    int64_t column, int32_t len)
{
    const event_t reusable = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(reusable);
    WaitFlag<HardEvent::V_MTE2>(reusable);
    if constexpr (std::is_same<ScalesT, float>::value) {
        LoadTile(scalesBuf_.Get<float>(), scalesGm_, column, len);
    } else {
        LoadTile(scalesRawBuf_.Get<ScalesT>(), scalesGm_, column, len);
    }
    if (hasZeroPoints_ != 0) {
        LoadTile(zpRawBuf_.Get<OffsetsT>(), zpGm_, column, len);
    }
    const event_t loaded = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(loaded);
    WaitFlag<HardEvent::MTE2_V>(loaded);
    if constexpr (!std::is_same<ScalesT, float>::value) {
        Cast(scalesBuf_.Get<float>(), scalesRawBuf_.Get<ScalesT>(), RoundMode::CAST_NONE, len);
    }
    if (hasZeroPoints_ != 0) {
        Cast(zpBuf_.Get<float>(), zpRawBuf_.Get<OffsetsT>(), RoundMode::CAST_NONE, len);
    }
    PipeBarrier<PIPE_V>();
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::QuantOneTile(
    int64_t srcOffset, int64_t dstOffset, int32_t len)
{
    LocalTensor<float> work = workBuf_.Get<float>();
    const event_t reusable = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(reusable);
    WaitFlag<HardEvent::V_MTE2>(reusable);
    LoadTile(updatesBuf_.Get<UpdatesT>(), updatesGm_, srcOffset, len);
    const event_t loaded = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(loaded);
    WaitFlag<HardEvent::MTE2_V>(loaded);
    Cast(work, updatesBuf_.Get<UpdatesT>(), RoundMode::CAST_NONE, len);
    PipeBarrier<PIPE_V>();
    if (reciprocalScale_ != 0) {
        Mul(work, work, scalesBuf_.Get<float>(), len);
    } else {
        Div(work, work, scalesBuf_.Get<float>(), len);
    }
    PipeBarrier<PIPE_V>();
    if (hasZeroPoints_ != 0) {
        Add(work, work, zpBuf_.Get<float>(), len);
        PipeBarrier<PIPE_V>();
    }
    Maxs(work, work, -128.0f, len);
    PipeBarrier<PIPE_V>();
    Mins(work, work, 127.0f, len);
    PipeBarrier<PIPE_V>();
    // Preserve the built-in path: never round to half before RINT.
    Cast(int16Buf_.Get<int16_t>(), work, RoundMode::CAST_RINT, len);
    PipeBarrier<PIPE_V>();
    Cast(halfBuf_.Get<half>(), int16Buf_.Get<int16_t>(), RoundMode::CAST_NONE, len);
    PipeBarrier<PIPE_V>();
    Cast(outBuf_.Get<int8_t>(), halfBuf_.Get<half>(), RoundMode::CAST_NONE, len);
    const event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(ready);
    WaitFlag<HardEvent::V_MTE3>(ready);
    DataCopyExtParams params = {1, static_cast<uint32_t>(len), 0, 0, 0};
    DataCopyPad(outGm_[dstOffset], outBuf_.Get<int8_t>(), params);
    const event_t stored = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(stored);
    WaitFlag<HardEvent::MTE3_V>(stored);
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::Process()
{
    CopyVarToOut();
    SyncAll(); // copy-only cores must finish before scatter writes
    if (batchLen_ <= 0 || batchStart_ >= indexElements_ || lastDim_ <= 0 || updateAxis_ <= 0) {
        return;
    }
    LoadIndices();
    const int64_t srcBsStride = midSize_ * updateAxis_ * lastDim_;
    const int64_t dstBsStride = midSize_ * varAxis_ * lastDim_;
    for (int64_t column = 0; column < lastDim_;) {
        const int64_t remaining = lastDim_ - column;
        const int32_t len = remaining > ubFactor_ ? static_cast<int32_t>(ubFactor_) :
            static_cast<int32_t>(remaining);
        LoadQuantParams(column, len);
        for (int64_t bi = 0; bi < batchLen_; ++bi) {
            const int64_t p = batchStart_ + bi;
            if (p >= indexElements_) {
                break;
            }
            const int64_t idx = ReadIndex(p);
            for (int64_t m = 0; m < midSize_; ++m) {
                for (int64_t k = 0; k < updateAxis_; ++k) {
                    const int64_t src = p * srcBsStride + (m * updateAxis_ + k) * lastDim_ + column;
                    const int64_t dst = p * dstBsStride + (m * varAxis_ + idx + k) * lastDim_ + column;
                    QuantOneTile(src, dst, len);
                }
            }
        }
        column += len;
    }
}

} // namespace NsQuantUpdateScatter
#endif // QUANTUPDATESCATTER_H
