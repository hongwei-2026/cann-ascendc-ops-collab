/*!
 * \file quant_update_scatter.h
 * \brief QuantUpdateScatter 算子 kernel（Q001 性能基线+）
 *
 * 语义对齐 samples/reference.py / 内置 AscendC：
 *   y = var.copy()
 *   for p, idx in enumerate(indices):
 *       for k in range(updates.shape[axis]):          # axis 仅支持 -2
 *           y[p, ..., idx+k, :] = cast_int8_rint(
 *               fp32(updates[p, ..., k, :]) / fp32(scales) + fp32(zero_points))
 *   quant_axis 仅支持 -1；除法在 fp32；CAST_RINT；结果 clip 到 [-128, 127]。
 *   reciprocal_scale=true 时改为乘法。
 *
 * Q001 相对 ZIP 朴素版：
 *   1) var→out 按核切分并行拷贝（inplace 同址跳过）
 *   2) 本核 batch 段 indices 一次载入 UB，避免 GetValue 标量读 GM
 *   3) scales/zp 按 lastDim tile 只载一次，跨所有行复用
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
constexpr int32_t INDICES_UB_CAP = 2048; // 每核最多缓存的 indices 个数

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
    __aicore__ inline void LoadIndices();
    __aicore__ inline void LoadScalesZp(int64_t dimOffset, int32_t len);
    __aicore__ inline void QuantOneRowTile(int64_t srcOffset, int64_t dstOffset, int64_t dimOffset, int32_t len);
    __aicore__ inline int64_t ReadIndexLocal(int64_t localP) const;

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
    int64_t coreNum_ = 1;
    int32_t bufElems_ = 64;
    int32_t indicesCap_ = 0;
    int32_t indicesCached_ = 0;
    uint64_t varAddr_ = 0;
    uint64_t outAddr_ = 0;
};

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::Init(GM_ADDR var, GM_ADDR indices,
    GM_ADDR updates, GM_ADDR quant_scales, GM_ADDR quant_zero_points, GM_ADDR var_out,
    const QuantUpdateScatterTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    midSize_ = tilingData->midSize > 0 ? tilingData->midSize : 1;
    updateAxis_ = tilingData->updateAxis;
    varAxis_ = tilingData->varAxis;
    lastDim_ = tilingData->lastDim;
    indexElements_ = tilingData->indexElements;
    hasZeroPoints_ = tilingData->hasZeroPoints;
    reciprocalScale_ = tilingData->reciprocalScale;
    ubFactor_ = tilingData->ubFactor > 0 ? tilingData->ubFactor : 64;
    coreNum_ = tilingData->coreNum > 0 ? tilingData->coreNum : 1;
    bufElems_ = AlignUpInt(static_cast<int32_t>(ubFactor_), VEC_ALIGN_FP32);

    const int64_t blockIdx = GetBlockIdx();
    const int64_t factor = tilingData->blockFactor > 0 ? tilingData->blockFactor : tilingData->batchSize;
    batchStart_ = blockIdx * factor;
    const int64_t remain = tilingData->batchSize - batchStart_;
    batchLen_ = remain > factor ? factor : (remain > 0 ? remain : 0);

    varAddr_ = reinterpret_cast<uint64_t>(var);
    outAddr_ = reinterpret_cast<uint64_t>(var_out);
    varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(var));
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(var_out));
    updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ UpdatesT*>(updates));
    scalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesT*>(quant_scales));
    zpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OffsetsT*>(quant_zero_points));
    indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IndicesT*>(indices));

    const int32_t n = bufElems_;
    indicesCap_ = INDICES_UB_CAP;
    if (batchLen_ > 0 && batchLen_ < indicesCap_) {
        indicesCap_ = static_cast<int32_t>(batchLen_);
    }
    if (indicesCap_ < 1) {
        indicesCap_ = 1;
    }
    indicesCap_ = AlignUpInt(indicesCap_, 8);

    pipe_.InitBuffer(updatesBuf_, n * static_cast<int32_t>(sizeof(UpdatesT)));
    pipe_.InitBuffer(scalesRawBuf_, n * 4);
    pipe_.InitBuffer(zpRawBuf_, n * 4);
    pipe_.InitBuffer(scalesBuf_, n * static_cast<int32_t>(sizeof(float)));
    pipe_.InitBuffer(zpBuf_, n * static_cast<int32_t>(sizeof(float)));
    pipe_.InitBuffer(workBuf_, n * static_cast<int32_t>(sizeof(float)));
    pipe_.InitBuffer(int16Buf_, n * static_cast<int32_t>(sizeof(int16_t)));
    pipe_.InitBuffer(halfBuf_, n * static_cast<int32_t>(sizeof(half)));
    pipe_.InitBuffer(outBuf_, n * static_cast<int32_t>(sizeof(int8_t)));
    pipe_.InitBuffer(indicesBuf_, indicesCap_ * static_cast<int32_t>(sizeof(IndicesT)));
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::CopyVarToOut()
{
    // inplace 同址跳过；多核按 totalNum 均分拷贝
    if (totalNum_ <= 0 || varAddr_ == outAddr_) {
        return;
    }
    const int64_t cores = coreNum_ > 0 ? coreNum_ : 1;
    const int64_t blockIdx = GetBlockIdx();
    if (blockIdx >= cores) {
        return;
    }
    const int64_t factor = (totalNum_ + cores - 1) / cores;
    const int64_t start = blockIdx * factor;
    if (start >= totalNum_) {
        return;
    }
    int64_t lenAll = totalNum_ - start;
    if (lenAll > factor) {
        lenAll = factor;
    }

    LocalTensor<int8_t> tmp = outBuf_.Get<int8_t>();
    const int32_t chunkCap = bufElems_;
    int64_t done = 0;
    while (done < lenAll) {
        int32_t len = static_cast<int32_t>(lenAll - done);
        if (len > chunkCap) {
            len = chunkCap;
        }
        constexpr int32_t blockElems = 32;
        const int32_t rem = len % blockElems;
        const uint8_t rightPad = (rem == 0) ? 0 : static_cast<uint8_t>(blockElems - rem);
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len), 0, 0, 0};
        DataCopyPadExtParams<int8_t> padParams = {true, 0, rightPad, 0};
        DataCopyPad(tmp, varGm_[start + done], copyParams, padParams);
        event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(e1);
        WaitFlag<HardEvent::MTE2_MTE3>(e1);
        DataCopyPad(outGm_[start + done], tmp, copyParams);
        event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(e2);
        WaitFlag<HardEvent::MTE3_MTE2>(e2);
        done += len;
    }
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::LoadIndices()
{
    indicesCached_ = 0;
    if (batchLen_ <= 0) {
        return;
    }
    LocalTensor<IndicesT> idxLocal = indicesBuf_.Get<IndicesT>();
    int32_t len = static_cast<int32_t>(batchLen_);
    if (len > indicesCap_) {
        len = indicesCap_;
    }
    constexpr int32_t blockElems = static_cast<int32_t>(32 / sizeof(IndicesT));
    const int32_t rem = len % blockElems;
    const uint8_t rightPad = (rem == 0) ? 0 : static_cast<uint8_t>(blockElems - rem);
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(IndicesT)), 0, 0, 0};
    DataCopyPadExtParams<IndicesT> padParams = {true, 0, rightPad, IndicesT(0)};
    DataCopyPad(idxLocal, indicesGm_[batchStart_], copyParams, padParams);
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e);
    WaitFlag<HardEvent::MTE2_S>(e);
    indicesCached_ = len;
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline int64_t QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::ReadIndexLocal(
    int64_t localP) const
{
    const int64_t p = batchStart_ + localP;
    if (p < 0 || p >= indexElements_) {
        return 0;
    }
    if (localP >= 0 && localP < indicesCached_) {
        LocalTensor<IndicesT> idxLocal = indicesBuf_.Get<IndicesT>();
        return static_cast<int64_t>(idxLocal.GetValue(static_cast<uint32_t>(localP)));
    }
    return static_cast<int64_t>(indicesGm_.GetValue(p));
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::LoadScalesZp(
    int64_t dimOffset, int32_t len)
{
    LocalTensor<float> scales = scalesBuf_.Get<float>();
    LocalTensor<float> zp = zpBuf_.Get<float>();

    if constexpr (std::is_same<ScalesT, float>::value) {
        constexpr int32_t blockElems = 8;
        const int32_t rem = len % blockElems;
        const uint8_t rightPad = (rem == 0) ? 0 : static_cast<uint8_t>(blockElems - rem);
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams = {true, 0, rightPad, 0.0f};
        DataCopyPad(scales, scalesGm_[dimOffset], copyParams, padParams);
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(e);
        WaitFlag<HardEvent::MTE2_V>(e);
    } else {
        LocalTensor<ScalesT> raw = scalesRawBuf_.Get<ScalesT>();
        constexpr int32_t blockElems = static_cast<int32_t>(32 / sizeof(ScalesT));
        const int32_t rem = len % blockElems;
        const uint8_t rightPad = (rem == 0) ? 0 : static_cast<uint8_t>(blockElems - rem);
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(ScalesT)), 0, 0, 0};
        DataCopyPadExtParams<ScalesT> padParams = {true, 0, rightPad, ScalesT(0)};
        DataCopyPad(raw, scalesGm_[dimOffset], copyParams, padParams);
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(e);
        WaitFlag<HardEvent::MTE2_V>(e);
        Cast(scales, raw, RoundMode::CAST_NONE, len);
    }

    if (hasZeroPoints_ != 0) {
        if constexpr (std::is_same<OffsetsT, int32_t>::value) {
            LocalTensor<int32_t> zpRaw = zpRawBuf_.Get<int32_t>();
            constexpr int32_t blockElems = 8;
            const int32_t rem = len % blockElems;
            const uint8_t rightPad = (rem == 0) ? 0 : static_cast<uint8_t>(blockElems - rem);
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams = {true, 0, rightPad, 0};
            DataCopyPad(zpRaw, zpGm_[dimOffset], copyParams, padParams);
            event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(e);
            WaitFlag<HardEvent::MTE2_V>(e);
            Cast(zp, zpRaw, RoundMode::CAST_NONE, len);
        } else {
            LocalTensor<OffsetsT> zpRaw = zpRawBuf_.Get<OffsetsT>();
            constexpr int32_t blockElems = static_cast<int32_t>(32 / sizeof(OffsetsT));
            const int32_t rem = len % blockElems;
            const uint8_t rightPad = (rem == 0) ? 0 : static_cast<uint8_t>(blockElems - rem);
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(OffsetsT)), 0, 0, 0};
            DataCopyPadExtParams<OffsetsT> padParams = {true, 0, rightPad, OffsetsT(0)};
            DataCopyPad(zpRaw, zpGm_[dimOffset], copyParams, padParams);
            event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(e);
            WaitFlag<HardEvent::MTE2_V>(e);
            Cast(zp, zpRaw, RoundMode::CAST_NONE, len);
        }
    }
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::QuantOneRowTile(
    int64_t srcOffset, int64_t dstOffset, int64_t dimOffset, int32_t len)
{
    LocalTensor<UpdatesT> updLocal = updatesBuf_.Get<UpdatesT>();
    LocalTensor<float> scales = scalesBuf_.Get<float>();
    LocalTensor<float> zp = zpBuf_.Get<float>();
    LocalTensor<float> work = workBuf_.Get<float>();
    LocalTensor<int16_t> i16Tmp = int16Buf_.Get<int16_t>();
    LocalTensor<half> halfTmp = halfBuf_.Get<half>();
    LocalTensor<int8_t> outLocal = outBuf_.Get<int8_t>();

    const int32_t calcCount = AlignUpInt(len, VEC_ALIGN_FP32);

    {
        constexpr int32_t blockElems = static_cast<int32_t>(32 / sizeof(UpdatesT));
        const int32_t rem = len % blockElems;
        const uint8_t rightPad = (rem == 0) ? 0 : static_cast<uint8_t>(blockElems - rem);
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(UpdatesT)), 0, 0, 0};
        DataCopyPadExtParams<UpdatesT> padParams = {true, 0, rightPad, UpdatesT(0)};
        DataCopyPad(updLocal, updatesGm_[srcOffset + dimOffset], copyParams, padParams);
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(e);
        WaitFlag<HardEvent::MTE2_V>(e);
        Cast(work, updLocal, RoundMode::CAST_NONE, len);
    }

    if (reciprocalScale_ != 0) {
        Mul(work, work, scales, calcCount);
    } else {
        Div(work, work, scales, calcCount);
    }
    if (hasZeroPoints_ != 0) {
        Add(work, work, zp, calcCount);
    }

    // 与内置一致：fp32 --RINT--> int16 --> half --> int8
    Maxs(work, work, -128.0f, calcCount);
    Mins(work, work, 127.0f, calcCount);
    Cast(i16Tmp, work, RoundMode::CAST_RINT, len);
    Cast(halfTmp, i16Tmp, RoundMode::CAST_NONE, len);
    Cast(outLocal, halfTmp, RoundMode::CAST_NONE, len);

    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(ev);
    WaitFlag<HardEvent::V_MTE3>(ev);
    DataCopyExtParams outParams = {1, static_cast<uint32_t>(len), 0, 0, 0};
    DataCopyPad(outGm_[dstOffset + dimOffset], outLocal, outParams);
    event_t em = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(em);
    WaitFlag<HardEvent::MTE3_V>(em);
}

template <typename UpdatesT, typename ScalesT, typename IndicesT, typename OffsetsT>
__aicore__ inline void QuantUpdateScatter<UpdatesT, ScalesT, IndicesT, OffsetsT>::Process()
{
    CopyVarToOut();
    SyncAll();

    if (batchLen_ <= 0 || lastDim_ <= 0) {
        return;
    }

    LoadIndices();

    const int64_t srcBsStride = midSize_ * updateAxis_ * lastDim_;
    const int64_t dstBsStride = midSize_ * varAxis_ * lastDim_;
    const int64_t srcMidStride = updateAxis_ * lastDim_;
    const int64_t dstMidStride = varAxis_ * lastDim_;

    // 外层按 lastDim tile：scales/zp 只载一次，内层扫所有待写行
    int64_t dimDone = 0;
    while (dimDone < lastDim_) {
        int32_t len = static_cast<int32_t>(lastDim_ - dimDone);
        if (len > static_cast<int32_t>(ubFactor_)) {
            len = static_cast<int32_t>(ubFactor_);
        }
        if (len > bufElems_) {
            len = bufElems_;
        }

        LoadScalesZp(dimDone, len);

        for (int64_t bi = 0; bi < batchLen_; ++bi) {
            const int64_t p = batchStart_ + bi;
            if (p >= indexElements_) {
                break;
            }
            const int64_t idx = ReadIndexLocal(bi);
            for (int64_t m = 0; m < midSize_; ++m) {
                for (int64_t k = 0; k < updateAxis_; ++k) {
                    const int64_t src = p * srcBsStride + m * srcMidStride + k * lastDim_;
                    const int64_t dst = p * dstBsStride + m * dstMidStride + (idx + k) * lastDim_;
                    QuantOneRowTile(src, dst, dimDone, len);
                }
            }
        }
        dimDone += len;
    }
}

} // namespace NsQuantUpdateScatter
#endif // QUANTUPDATESCATTER_H
