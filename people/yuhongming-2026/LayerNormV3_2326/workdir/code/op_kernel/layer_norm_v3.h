/*!
 * \file layer_norm_v3.h
 * \brief LayerNormV3：按行两遍归约，float32 计算，IO 支持 float/half/bf16。
 *
 * y = (x - mean) * rstd * gamma + beta
 * mean/var 在 begin_norm_axis 及其后的维度上计算（总体方差，除以 N）。
 * gamma/beta 与 x 的尾维对齐：扁平下标 g 使用 gamma[g % gammaNum]。
 */

#ifndef LAYERNORMV3_H
#define LAYERNORMV3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "layer_norm_v3_tiling_data.h"
#include "layer_norm_v3_tiling_key.h"

namespace NsLayerNormV3 {

using namespace AscendC;

template <typename T>
class LayerNormV3 {
public:
    __aicore__ inline LayerNormV3() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
        const LayerNormV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline float BitsToFloat(int64_t bits) const;
    __aicore__ inline void LoadAsFloat(const GlobalTensor<T>& gm, int64_t offset, const LocalTensor<float>& dst,
        int64_t valid);
    __aicore__ inline void StoreFloat(const GlobalTensor<T>& gm, int64_t offset, const LocalTensor<float>& src,
        int64_t valid);
    __aicore__ inline void StoreScalar(const GlobalTensor<T>& gm, int64_t index, float value);
    __aicore__ inline float SumAll(const LocalTensor<float>& data);
    __aicore__ inline float RowMean(int64_t row);
    __aicore__ inline float RowRstd(int64_t row, float mean);
    __aicore__ inline void WriteAffine(int64_t row, float mean, float rstd);
    __aicore__ inline int64_t NextValid(int64_t row, int64_t colOff, int64_t& gpos) const;

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> bufX_;
    TBuf<QuePosition::VECCALC> bufTmp_;
    TBuf<QuePosition::VECCALC> bufG_;
    TBuf<QuePosition::VECCALC> bufB_;
    TBuf<QuePosition::VECCALC> bufWork_;
    TBuf<QuePosition::VECCALC> bufRaw_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> gammaGm_;
    GlobalTensor<T> betaGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> meanGm_;
    GlobalTensor<T> rstdGm_;

    int64_t rowNum_ = 0;
    int64_t colNum_ = 0;
    int64_t gammaNum_ = 0;
    int64_t tileLength_ = 0;
    int64_t tileAlign_ = 64;
    int64_t rowBegin_ = 0;
    int64_t rowEnd_ = 0;
    float epsilon_ = 1e-5f;
    float gammaScalar_ = 1.0f;
    float betaScalar_ = 0.0f;
    bool scalarAffine_ = false;
    bool active_ = false;
};

template <typename T>
__aicore__ inline float LayerNormV3<T>::BitsToFloat(int64_t bits) const
{
    union {
        uint32_t u;
        float f;
    } v;
    v.u = static_cast<uint32_t>(bits);
    return v.f;
}

template <typename T>
__aicore__ inline void LayerNormV3<T>::Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
    GM_ADDR rstd, const LayerNormV3TilingData* tilingData)
{
    rowNum_ = tilingData->rowNum;
    colNum_ = tilingData->colNum;
    gammaNum_ = tilingData->gammaNum;
    tileLength_ = tilingData->tileLength;
    tileAlign_ = tilingData->tileAlign;
    epsilon_ = BitsToFloat(tilingData->epsilonBits);
    scalarAffine_ = (gammaNum_ == 1);

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= tilingData->usedCores || rowNum_ <= 0 || colNum_ <= 0 || gammaNum_ <= 0 || tileAlign_ < 64 ||
        tileLength_ <= 0) {
        active_ = false;
        return;
    }
    int64_t rowCount = tilingData->rowsPerCore;
    if (blockIdx < tilingData->rowRemain) {
        rowCount = tilingData->rowsPerCore + 1;
        rowBegin_ = blockIdx * rowCount;
    } else {
        rowBegin_ = tilingData->rowRemain * (tilingData->rowsPerCore + 1) +
                    (blockIdx - tilingData->rowRemain) * tilingData->rowsPerCore;
    }
    rowEnd_ = rowBegin_ + rowCount;
    if (rowBegin_ >= rowNum_ || rowCount <= 0) {
        active_ = false;
        return;
    }
    if (rowEnd_ > rowNum_) {
        rowEnd_ = rowNum_;
    }
    active_ = true;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), rowNum_ * colNum_);
    gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma), gammaNum_);
    betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta), gammaNum_);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), rowNum_ * colNum_);
    meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(mean), rowNum_);
    rstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rstd), rowNum_);

    const uint32_t bytes = static_cast<uint32_t>(tileAlign_ * static_cast<int64_t>(sizeof(float)));
    pipe_.InitBuffer(bufX_, bytes);
    pipe_.InitBuffer(bufTmp_, bytes);
    pipe_.InitBuffer(bufG_, bytes);
    pipe_.InitBuffer(bufB_, bytes);
    pipe_.InitBuffer(bufWork_, bytes);
    pipe_.InitBuffer(bufRaw_, bytes);
}

template <typename T>
__aicore__ inline void LayerNormV3<T>::LoadAsFloat(const GlobalTensor<T>& gm, int64_t offset,
    const LocalTensor<float>& dst, int64_t valid)
{
    Duplicate(dst, 0.0f, static_cast<uint32_t>(tileAlign_));
    PipeBarrier<PIPE_ALL>();
    if (valid <= 0) {
        return;
    }
    if constexpr (sizeof(T) == sizeof(float)) {
        DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(dst, gm[offset], p, pad);
    } else {
        LocalTensor<float> rawFloat = bufRaw_.Get<float>();
        Duplicate(rawFloat, 0.0f, static_cast<uint32_t>(tileAlign_));
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T> raw = bufRaw_.Get<T>();
        DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(raw, gm[offset], p, pad);
        PipeBarrier<PIPE_ALL>();
        Cast(dst, raw, RoundMode::CAST_NONE, static_cast<uint32_t>(tileAlign_));
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void LayerNormV3<T>::StoreFloat(const GlobalTensor<T>& gm, int64_t offset,
    const LocalTensor<float>& src, int64_t valid)
{
    if (valid <= 0) {
        return;
    }
    if constexpr (sizeof(T) == sizeof(float)) {
        DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
        DataCopyPad(gm[offset], src, p);
    } else {
        LocalTensor<T> raw = bufRaw_.Get<T>();
        Cast(raw, src, RoundMode::CAST_RINT, static_cast<uint32_t>(tileAlign_));
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(gm[offset], raw, p);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void LayerNormV3<T>::StoreScalar(const GlobalTensor<T>& gm, int64_t index, float value)
{
    LocalTensor<float> slot = bufWork_.Get<float>();
    Duplicate(slot, 0.0f, 64);
    slot.SetValue(0, value);
    PipeBarrier<PIPE_ALL>();
    if constexpr (sizeof(T) == sizeof(float)) {
        DataCopyExtParams p{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(gm[index], slot, p);
    } else {
        LocalTensor<T> raw = bufRaw_.Get<T>();
        Cast(raw, slot, RoundMode::CAST_RINT, 64);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(gm[index], raw, p);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline float LayerNormV3<T>::SumAll(const LocalTensor<float>& data)
{
    LocalTensor<float> scratch = bufWork_.Get<float>();
    LocalTensor<float> cur = data;
    LocalTensor<float> nxt = scratch;
    int32_t n = static_cast<int32_t>(tileAlign_);
    while (n > 64) {
        if ((n & 1) != 0) {
            PipeBarrier<PIPE_ALL>();
            const float extra = cur.GetValue(n - 1);
            cur.SetValue(0, cur.GetValue(0) + extra);
            n -= 1;
            PipeBarrier<PIPE_ALL>();
        }
        const int32_t half = n / 2;
        Add(nxt, cur, cur[half], static_cast<uint32_t>(half));
        PipeBarrier<PIPE_V>();
        LocalTensor<float> swap = cur;
        cur = nxt;
        nxt = swap;
        n = half;
    }
    PipeBarrier<PIPE_ALL>();
    float sum = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        sum += cur.GetValue(i);
    }
    return sum;
}

template <typename T>
__aicore__ inline int64_t LayerNormV3<T>::NextValid(int64_t row, int64_t colOff, int64_t& gpos) const
{
    int64_t valid = tileLength_;
    const int64_t remain = colNum_ - colOff;
    if (valid > remain) {
        valid = remain;
    }
    if (scalarAffine_) {
        gpos = 0;
        return valid;
    }
    const int64_t flat = row * colNum_ + colOff;
    gpos = flat % gammaNum_;
    const int64_t room = gammaNum_ - gpos;
    if (valid > room) {
        valid = room;
    }
    return valid;
}

template <typename T>
__aicore__ inline float LayerNormV3<T>::RowMean(int64_t row)
{
    LocalTensor<float> xLocal = bufX_.Get<float>();
    float sum = 0.0f;
    int64_t colOff = 0;
    while (colOff < colNum_) {
        int64_t valid = tileLength_;
        const int64_t remain = colNum_ - colOff;
        if (valid > remain) {
            valid = remain;
        }
        LoadAsFloat(xGm_, row * colNum_ + colOff, xLocal, valid);
        sum += SumAll(xLocal);
        colOff += valid;
    }
    return sum / static_cast<float>(colNum_);
}

template <typename T>
__aicore__ inline float LayerNormV3<T>::RowRstd(int64_t row, float mean)
{
    LocalTensor<float> xLocal = bufX_.Get<float>();
    LocalTensor<float> tmp = bufTmp_.Get<float>();
    float sumsq = 0.0f;
    int64_t colOff = 0;
    while (colOff < colNum_) {
        int64_t valid = tileLength_;
        const int64_t remain = colNum_ - colOff;
        if (valid > remain) {
            valid = remain;
        }
        LoadAsFloat(xGm_, row * colNum_ + colOff, xLocal, valid);
        Adds(tmp, xLocal, -mean, static_cast<uint32_t>(tileAlign_));
        PipeBarrier<PIPE_V>();
        Mul(xLocal, tmp, tmp, static_cast<uint32_t>(tileAlign_));
        PipeBarrier<PIPE_V>();
        sumsq += SumAll(xLocal);
        colOff += valid;
    }
    const float var = sumsq / static_cast<float>(colNum_);
    LocalTensor<float> slot = bufWork_.Get<float>();
    LocalTensor<float> rstdLocal = bufTmp_.Get<float>();
    Duplicate(slot, 0.0f, 64);
    slot.SetValue(0, var + epsilon_);
    PipeBarrier<PIPE_ALL>();
    Rsqrt(rstdLocal, slot, 64);
    PipeBarrier<PIPE_ALL>();
    return rstdLocal.GetValue(0);
}

template <typename T>
__aicore__ inline void LayerNormV3<T>::WriteAffine(int64_t row, float mean, float rstd)
{
    LocalTensor<float> xLocal = bufX_.Get<float>();
    LocalTensor<float> tmp = bufTmp_.Get<float>();
    LocalTensor<float> gLocal = bufG_.Get<float>();
    LocalTensor<float> bLocal = bufB_.Get<float>();
    int64_t colOff = 0;
    while (colOff < colNum_) {
        int64_t gpos = 0;
        const int64_t valid = NextValid(row, colOff, gpos);
        LoadAsFloat(xGm_, row * colNum_ + colOff, xLocal, valid);
        if (scalarAffine_) {
            Duplicate(gLocal, gammaScalar_, static_cast<uint32_t>(tileAlign_));
            Duplicate(bLocal, betaScalar_, static_cast<uint32_t>(tileAlign_));
            PipeBarrier<PIPE_V>();
        } else {
            LoadAsFloat(gammaGm_, gpos, gLocal, valid);
            LoadAsFloat(betaGm_, gpos, bLocal, valid);
        }
        Adds(tmp, xLocal, -mean, static_cast<uint32_t>(tileAlign_));
        PipeBarrier<PIPE_V>();
        Muls(xLocal, tmp, rstd, static_cast<uint32_t>(tileAlign_));
        PipeBarrier<PIPE_V>();
        Mul(tmp, xLocal, gLocal, static_cast<uint32_t>(tileAlign_));
        PipeBarrier<PIPE_V>();
        Add(xLocal, tmp, bLocal, static_cast<uint32_t>(tileAlign_));
        PipeBarrier<PIPE_ALL>();
        StoreFloat(yGm_, row * colNum_ + colOff, xLocal, valid);
        colOff += valid;
    }
}

template <typename T>
__aicore__ inline void LayerNormV3<T>::Process()
{
    if (!active_) {
        return;
    }
    if (scalarAffine_) {
        LocalTensor<float> slot = bufX_.Get<float>();
        LoadAsFloat(gammaGm_, 0, slot, 1);
        gammaScalar_ = slot.GetValue(0);
        LoadAsFloat(betaGm_, 0, slot, 1);
        betaScalar_ = slot.GetValue(0);
    }
    for (int64_t row = rowBegin_; row < rowEnd_; ++row) {
        const float mean = RowMean(row);
        const float rstd = RowRstd(row, mean);
        StoreScalar(meanGm_, row, mean);
        StoreScalar(rstdGm_, row, rstd);
        WriteAffine(row, mean, rstd);
    }
}

} // namespace NsLayerNormV3
#endif // LAYERNORMV3_H
