/*!
 * \file sync_batch_norm_gather_stats.h
 * \brief SyncBatchNormGatherStats 算子 kernel
 *
 * 判题 / samples/reference.py 语义（与 910B 上插 Cast 后的实测一致）：
 *   所有浮点输入先落到 fp16 网格，中间用 fp32：
 *     batch_mean   = sum(total_sum) / n
 *     batch_var    = sum(total_square_sum) / n - batch_mean^2
 *     batch_invstd = 1 / sqrt(batch_var + eps)
 *     new_mean     = (1 - momentum) * mean + momentum * batch_mean
 *     new_var      = n/(n-1) * momentum * batch_var + (1 - momentum) * variance
 *   四个输出再按 fp16 舍入后写出；n == 1 时无偏系数为 +inf。
 *   total_sum 按 [world_size, C] 沿设备维累加。
 */

#ifndef SYNCBATCHNORMGATHERSTATS_H
#define SYNCBATCHNORMGATHERSTATS_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sync_batch_norm_gather_stats_tiling_data.h"
#include "sync_batch_norm_gather_stats_tiling_key.h"

namespace NsSyncBatchNormGatherStats {

using namespace AscendC;

constexpr int32_t VEC_ALIGN = 8; // fp32 32B 对齐

__aicore__ inline int32_t AlignUpInt(int32_t value, int32_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

__aicore__ inline float BitsToFloat(int64_t bits)
{
    uint32_t u = static_cast<uint32_t>(bits);
    float value = 0.0f;
    *reinterpret_cast<uint32_t*>(&value) = u;
    return value;
}

__aicore__ inline float PositiveInf()
{
    uint32_t u = 0x7f800000u;
    float value = 0.0f;
    *reinterpret_cast<uint32_t*>(&value) = u;
    return value;
}

__aicore__ inline float QuietNaN()
{
    uint32_t u = 0x7fc00000u;
    float value = 0.0f;
    *reinterpret_cast<uint32_t*>(&value) = u;
    return value;
}

template <typename T>
class SyncBatchNormGatherStats {
public:
    __aicore__ inline SyncBatchNormGatherStats() {}

    __aicore__ inline void Init(GM_ADDR totalSum, GM_ADDR totalSquareSum, GM_ADDR sampleCount, GM_ADDR mean,
        GM_ADDR variance, GM_ADDR batchMean, GM_ADDR batchInvstd, GM_ADDR newMean, GM_ADDR newVariance,
        const SyncBatchNormGatherStatsTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline float SumSampleCount() const;
    __aicore__ inline void QuantFp16Vec(LocalTensor<float>& data, int32_t len);
    __aicore__ inline void LoadQuantized(const GlobalTensor<T>& gm, int64_t offset, int32_t len, int32_t calcCount,
        LocalTensor<float>& dst);
    __aicore__ inline void StoreAsInputDtype(GlobalTensor<T>& gm, int64_t offset, int32_t len, LocalTensor<float>& src);
    __aicore__ inline void StoreFloat(GlobalTensor<float>& gm, int64_t offset, int32_t len, LocalTensor<float>& src);
    __aicore__ inline void ComputeTile(int64_t channel, int32_t len);

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> loadBuf_;
    TBuf<TPosition::VECCALC> accSumBuf_;
    TBuf<TPosition::VECCALC> accSqBuf_;
    TBuf<TPosition::VECCALC> meanBuf_;
    TBuf<TPosition::VECCALC> varBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    GlobalTensor<T> sumGm_;
    GlobalTensor<T> sqGm_;
    GlobalTensor<T> meanGm_;
    GlobalTensor<T> varGm_;
    GlobalTensor<T> batchMeanGm_;
    GlobalTensor<T> batchInvGm_;
    GlobalTensor<float> newMeanGm_;
    GlobalTensor<float> newVarGm_;
    GlobalTensor<int32_t> count32Gm_;
    GlobalTensor<int64_t> count64Gm_;

    int64_t channels_ = 0;
    int64_t blockStart_ = 0;
    int64_t blockLength_ = 0;
    int64_t worldSize_ = 1;
    int64_t countNum_ = 1;
    int64_t countIsInt64_ = 0;
    int64_t ubFactor_ = 64;
    int32_t bufElems_ = 64;
    float momentum_ = 0.1f;
    float eps_ = 1e-5f;
    float sampleCount_ = 0.0f;
};

template <typename T>
__aicore__ inline void SyncBatchNormGatherStats<T>::Init(GM_ADDR totalSum, GM_ADDR totalSquareSum, GM_ADDR sampleCount,
    GM_ADDR mean, GM_ADDR variance, GM_ADDR batchMean, GM_ADDR batchInvstd, GM_ADDR newMean, GM_ADDR newVariance,
    const SyncBatchNormGatherStatsTilingData* tilingData)
{
    channels_ = tilingData->totalNum;
    worldSize_ = tilingData->worldSize > 0 ? tilingData->worldSize : 1;
    countNum_ = tilingData->countNum > 0 ? tilingData->countNum : 0;
    countIsInt64_ = tilingData->countIsInt64;
    ubFactor_ = tilingData->ubFactor > 0 ? tilingData->ubFactor : 64;
    momentum_ = BitsToFloat(tilingData->momentumBits);
    eps_ = BitsToFloat(tilingData->epsBits);
    bufElems_ = AlignUpInt(static_cast<int32_t>(ubFactor_), VEC_ALIGN);

    const int64_t blockIdx = GetBlockIdx();
    const int64_t factor = tilingData->blockFactor > 0 ? tilingData->blockFactor : channels_;
    blockStart_ = blockIdx * factor;
    const int64_t remain = channels_ - blockStart_;
    blockLength_ = remain > factor ? factor : (remain > 0 ? remain : 0);

    sumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(totalSum));
    sqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(totalSquareSum));
    meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(mean));
    varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(variance));
    batchMeanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(batchMean));
    batchInvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(batchInvstd));
    newMeanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(newMean));
    newVarGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(newVariance));
    count32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sampleCount));
    count64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(sampleCount));

    const int32_t floatBytes = bufElems_ * static_cast<int32_t>(sizeof(float));
    pipe_.InitBuffer(rawBuf_, floatBytes);
    pipe_.InitBuffer(loadBuf_, floatBytes);
    pipe_.InitBuffer(accSumBuf_, floatBytes);
    pipe_.InitBuffer(accSqBuf_, floatBytes);
    pipe_.InitBuffer(meanBuf_, floatBytes);
    pipe_.InitBuffer(varBuf_, floatBytes);
    pipe_.InitBuffer(tmpBuf_, floatBytes);
}

template <typename T>
__aicore__ inline float SyncBatchNormGatherStats<T>::SumSampleCount() const
{
    int64_t acc = 0;
    if (countIsInt64_ != 0) {
        for (int64_t i = 0; i < countNum_; ++i) {
            acc += count64Gm_.GetValue(i);
        }
    } else {
        for (int64_t i = 0; i < countNum_; ++i) {
            acc += static_cast<int64_t>(count32Gm_.GetValue(i));
        }
    }
    return static_cast<float>(acc);
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStats<T>::QuantFp16Vec(LocalTensor<float>& data, int32_t len)
{
    LocalTensor<half> raw = rawBuf_.Get<half>();
    Cast(raw, data, RoundMode::CAST_RINT, len);
    Cast(data, raw, RoundMode::CAST_NONE, len);
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStats<T>::LoadQuantized(const GlobalTensor<T>& gm, int64_t offset,
    int32_t len, int32_t calcCount, LocalTensor<float>& dst)
{
    constexpr int32_t blockElems = static_cast<int32_t>(32 / sizeof(T));
    const int32_t rem = len % blockElems;
    const uint8_t rightPad = (rem == 0) ? 0 : static_cast<uint8_t>(blockElems - rem);

    Duplicate(dst, 0.0f, calcCount);

    if constexpr (std::is_same<T, float>::value) {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams = {true, 0, rightPad, 0.0f};
        event_t eventV2Mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventV2Mte);
        WaitFlag<HardEvent::V_MTE2>(eventV2Mte);
        DataCopyPad(dst, gm[offset], copyParams, padParams);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        // fp32 输入先落到 fp16 网格（判题 / reference 语义）
        QuantFp16Vec(dst, len);
    } else {
        LocalTensor<half> raw = rawBuf_.Get<half>();
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(half)), 0, 0, 0};
        DataCopyPadExtParams<half> padParams = {true, 0, rightPad, static_cast<half>(0.0f)};
        event_t eventV2Mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventV2Mte);
        WaitFlag<HardEvent::V_MTE2>(eventV2Mte);
        DataCopyPad(raw, gm[offset], copyParams, padParams);
        event_t eventMte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte);
        WaitFlag<HardEvent::MTE2_V>(eventMte);
        Cast(dst, raw, RoundMode::CAST_NONE, len);
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStats<T>::StoreAsInputDtype(GlobalTensor<T>& gm, int64_t offset, int32_t len,
    LocalTensor<float>& src)
{
    if constexpr (std::is_same<T, float>::value) {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPad(gm[offset], src, copyParams);
    } else {
        LocalTensor<half> raw = rawBuf_.Get<half>();
        Cast(raw, src, RoundMode::CAST_RINT, len);
        event_t eventV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventV);
        WaitFlag<HardEvent::V_MTE3>(eventV);
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(half)), 0, 0, 0};
        DataCopyPad(gm[offset], raw, copyParams);
        event_t eventM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventM);
        WaitFlag<HardEvent::MTE3_V>(eventM);
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStats<T>::StoreFloat(GlobalTensor<float>& gm, int64_t offset, int32_t len,
    LocalTensor<float>& src)
{
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    DataCopyPad(gm[offset], src, copyParams);
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStats<T>::ComputeTile(int64_t channel, int32_t len)
{
    const int32_t calcCount = AlignUpInt(len, VEC_ALIGN);
    LocalTensor<float> load = loadBuf_.Get<float>();
    LocalTensor<float> accSum = accSumBuf_.Get<float>();
    LocalTensor<float> accSq = accSqBuf_.Get<float>();
    LocalTensor<float> mean = meanBuf_.Get<float>();
    LocalTensor<float> var = varBuf_.Get<float>();
    LocalTensor<float> tmp = tmpBuf_.Get<float>();

    Duplicate(accSum, 0.0f, calcCount);
    Duplicate(accSq, 0.0f, calcCount);

    for (int64_t device = 0; device < worldSize_; ++device) {
        const int64_t offset = device * channels_ + channel;
        LoadQuantized(sumGm_, offset, len, calcCount, load);
        Add(accSum, accSum, load, calcCount);
        LoadQuantized(sqGm_, offset, len, calcCount, load);
        Add(accSq, accSq, load, calcCount);
    }
    LoadQuantized(meanGm_, channel, len, calcCount, mean);
    LoadQuantized(varGm_, channel, len, calcCount, var);

    const float n = sampleCount_;
    const float oneMinus = 1.0f - momentum_;
    const float unbias = (n == 1.0f) ? PositiveInf() : (n / (n - 1.0f));
    const float unbiasMom = unbias * momentum_;

    Duplicate(load, n, calcCount);
    Div(accSum, accSum, load, calcCount);
    Div(accSq, accSq, load, calcCount);
    Mul(tmp, accSum, accSum, calcCount);
    Sub(accSq, accSq, tmp, calcCount);
    // invstd：向量 Sqrt+Div（负 var+eps 在量化后再标量补 NaN，避免 Cast 清掉 NaN）
    Adds(tmp, accSq, eps_, calcCount);
    Sqrt(tmp, tmp, calcCount);
    Duplicate(load, 1.0f, calcCount);
    Div(tmp, load, tmp, calcCount);

    Muls(mean, mean, oneMinus, calcCount);
    Muls(load, accSum, momentum_, calcCount);
    Add(mean, mean, load, calcCount);

    Muls(var, var, oneMinus, calcCount);
    Muls(load, accSq, unbiasMom, calcCount);
    Add(var, var, load, calcCount);

    // 四个输出都落到 fp16 网格（fp32 用例以 float 写出 fp16 值；fp16 用例 Cast 写出）
    QuantFp16Vec(accSum, len);
    QuantFp16Vec(tmp, len);
    QuantFp16Vec(mean, len);
    QuantFp16Vec(var, len);

    // CPU 模拟：Sqrt(负)→0，且 Quant Cast 会吞掉 NaN，必须在量化之后按 batch_var 回写
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
    for (int32_t i = 0; i < len; ++i) {
        const float varEps = accSq.GetValue(i) + eps_;
        if (varEps < 0.0f) {
            tmp.SetValue(i, QuietNaN());
        } else if (varEps == 0.0f) {
            const float cur = tmp.GetValue(i);
            if (cur == 0.0f) {
                tmp.SetValue(i, PositiveInf());
            }
        }
    }
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventSV);
    WaitFlag<HardEvent::S_V>(eventSV);

    event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3);

    StoreAsInputDtype(batchMeanGm_, channel, len, accSum);
    StoreAsInputDtype(batchInvGm_, channel, len, tmp);
    StoreFloat(newMeanGm_, channel, len, mean);
    StoreFloat(newVarGm_, channel, len, var);

    event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMTE3V);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3V);
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStats<T>::Process()
{
    if (blockLength_ <= 0 || channels_ <= 0) {
        return;
    }
    sampleCount_ = SumSampleCount();
    int64_t done = 0;
    while (done < blockLength_) {
        int32_t len = static_cast<int32_t>(blockLength_ - done);
        if (len > static_cast<int32_t>(ubFactor_)) {
            len = static_cast<int32_t>(ubFactor_);
        }
        if (len > bufElems_) {
            len = bufElems_;
        }
        ComputeTile(blockStart_ + done, len);
        done += len;
    }
}

} // namespace NsSyncBatchNormGatherStats
#endif // SYNCBATCHNORMGATHERSTATS_H
