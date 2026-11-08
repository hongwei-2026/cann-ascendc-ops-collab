/*!
 * \file layer_norm_v3.h
 * \brief LayerNormV3：按行两遍归约，float32 计算。
 *
 * y = (x - mean) * rstd * gamma + beta
 * mean/var 在 begin_norm_axis 及其后的维度上计算（总体方差，除以 N）。
 * gamma/beta/mean/rstd 的类型是 P，x/y 的类型是 T。P 可以是 T，也可以是 float。
 * gamma/beta 与 x 从 begin_params_axis 起的尾部对齐：扁平下标 g 使用 gamma[g % gammaNum]。
 */

#ifndef LAYERNORMV3_H
#define LAYERNORMV3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "layer_norm_v3_tiling_data.h"
#include "layer_norm_v3_tiling_key.h"

namespace NsLayerNormV3 {

using namespace AscendC;

template <typename T, typename P>
class LayerNormV3 {
public:
    __aicore__ inline LayerNormV3() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
        const LayerNormV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename U>
    __aicore__ inline void LoadAsFloat(const GlobalTensor<U>& gm, int64_t offset, const LocalTensor<float>& dst,
        int64_t valid);
    template <typename U>
    __aicore__ inline void StoreFloat(const GlobalTensor<U>& gm, int64_t offset, const LocalTensor<float>& src,
        int64_t valid);
    template <typename U>
    __aicore__ inline void StoreScalar(const GlobalTensor<U>& gm, int64_t index, float value);
    __aicore__ inline void ZeroFrom(const LocalTensor<float>& dst, int64_t valid);
    __aicore__ inline float BitsToFloat(int64_t bits) const;
    __aicore__ inline float SumAll(const LocalTensor<float>& data);
    __aicore__ inline float ScalarRsqrt(float x) const;
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
    GlobalTensor<P> gammaGm_;
    GlobalTensor<P> betaGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<P> meanGm_;
    GlobalTensor<P> rstdGm_;

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

template <typename T, typename P>
__aicore__ inline float LayerNormV3<T, P>::BitsToFloat(int64_t bits) const
{
    union {
        uint32_t u;
        float f;
    } v;
    v.u = static_cast<uint32_t>(bits);
    return v.f;
}

template <typename T, typename P>
__aicore__ inline void LayerNormV3<T, P>::ZeroFrom(const LocalTensor<float>& dst, int64_t valid)
{
    if (valid >= tileAlign_) {
        return;
    }
    if (valid < 0) {
        valid = 0;
    }
    PipeBarrier<PIPE_ALL>();
    for (int64_t i = valid; i < tileAlign_; ++i) {
        dst.SetValue(static_cast<int32_t>(i), 0.0f);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T, typename P>
__aicore__ inline void LayerNormV3<T, P>::Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
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
    gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(gamma), gammaNum_);
    betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(beta), gammaNum_);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), rowNum_ * colNum_);
    meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(mean), rowNum_);
    rstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(rstd), rowNum_);

    const uint32_t bytes = static_cast<uint32_t>(tileAlign_ * static_cast<int64_t>(sizeof(float)));
    pipe_.InitBuffer(bufX_, bytes);
    pipe_.InitBuffer(bufTmp_, bytes);
    pipe_.InitBuffer(bufG_, bytes);
    pipe_.InitBuffer(bufB_, bytes);
    pipe_.InitBuffer(bufWork_, bytes);
    pipe_.InitBuffer(bufRaw_, bytes);
}

template <typename T, typename P>
template <typename U>
__aicore__ inline void LayerNormV3<T, P>::LoadAsFloat(const GlobalTensor<U>& gm, int64_t offset,
    const LocalTensor<float>& dst, int64_t valid)
{
    Duplicate(dst, 0.0f, static_cast<uint32_t>(tileAlign_));
    PipeBarrier<PIPE_ALL>();
    if (valid > 0) {
        if constexpr (sizeof(U) == sizeof(float)) {
            DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            DataCopyPad(dst, gm[offset], p, pad);
        } else {
            LocalTensor<float> rawFloat = bufRaw_.Get<float>();
            Duplicate(rawFloat, 0.0f, static_cast<uint32_t>(tileAlign_));
            PipeBarrier<PIPE_ALL>();
            LocalTensor<U> raw = bufRaw_.Get<U>();
            DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(U))), 0, 0, 0};
            DataCopyPadExtParams<U> pad{false, 0, 0, static_cast<U>(0)};
            DataCopyPad(raw, gm[offset], p, pad);
            PipeBarrier<PIPE_ALL>();
            Cast(dst, raw, RoundMode::CAST_NONE, static_cast<uint32_t>(tileAlign_));
        }
        PipeBarrier<PIPE_ALL>();
    }
    // DataCopyPad 会按 32 字节向上取整，多出来的元素不能进归约。
    ZeroFrom(dst, valid);
}

template <typename T, typename P>
template <typename U>
__aicore__ inline void LayerNormV3<T, P>::StoreFloat(const GlobalTensor<U>& gm, int64_t offset,
    const LocalTensor<float>& src, int64_t valid)
{
    if (valid <= 0) {
        return;
    }
    if constexpr (sizeof(U) == sizeof(float)) {
        DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
        DataCopyPad(gm[offset], src, p);
    } else {
        LocalTensor<U> raw = bufRaw_.Get<U>();
        Cast(raw, src, RoundMode::CAST_RINT, static_cast<uint32_t>(tileAlign_));
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(U))), 0, 0, 0};
        DataCopyPad(gm[offset], raw, p);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T, typename P>
template <typename U>
__aicore__ inline void LayerNormV3<T, P>::StoreScalar(const GlobalTensor<U>& gm, int64_t index, float value)
{
    LocalTensor<float> slot = bufWork_.Get<float>();
    Duplicate(slot, 0.0f, 64);
    slot.SetValue(0, value);
    PipeBarrier<PIPE_ALL>();
    if constexpr (sizeof(U) == sizeof(float)) {
        DataCopyExtParams p{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(gm[index], slot, p);
    } else {
        LocalTensor<U> raw = bufRaw_.Get<U>();
        Cast(raw, slot, RoundMode::CAST_RINT, 64);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, static_cast<uint32_t>(sizeof(U)), 0, 0, 0};
        DataCopyPad(gm[index], raw, p);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T, typename P>
__aicore__ inline float LayerNormV3<T, P>::SumAll(const LocalTensor<float>& data)
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

template <typename T, typename P>
__aicore__ inline float LayerNormV3<T, P>::ScalarRsqrt(float x) const
{
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    v.u = 0x5f3759dfu - (v.u >> 1);
    float y = v.f;
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

template <typename T, typename P>
__aicore__ inline int64_t LayerNormV3<T, P>::NextValid(int64_t row, int64_t colOff, int64_t& gpos) const
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

template <typename T, typename P>
__aicore__ inline float LayerNormV3<T, P>::RowMean(int64_t row)
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

template <typename T, typename P>
__aicore__ inline float LayerNormV3<T, P>::RowRstd(int64_t row, float mean)
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
        PipeBarrier<PIPE_ALL>();
        // 补零在减均值再平方后变成 mean^2，不能算进方差。
        ZeroFrom(xLocal, valid);
        sumsq += SumAll(xLocal);
        colOff += valid;
    }
    const float var = sumsq / static_cast<float>(colNum_);
    return ScalarRsqrt(var + epsilon_);
}

template <typename T, typename P>
__aicore__ inline void LayerNormV3<T, P>::WriteAffine(int64_t row, float mean, float rstd)
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

template <typename T, typename P>
__aicore__ inline void LayerNormV3<T, P>::Process()
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
