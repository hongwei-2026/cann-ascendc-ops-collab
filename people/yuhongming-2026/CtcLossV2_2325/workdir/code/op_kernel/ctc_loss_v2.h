/*!
 * \file ctc_loss_v2.h
 * \brief CtcLossV2 kernel：复刻 ops_legacy TIK 语义（samples/reference.py）
 */

#ifndef CTCLOSSV2_H
#define CTCLOSSV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ctc_loss_v2_tiling_data.h"
#include "ctc_loss_v2_tiling_key.h"

namespace NsCtcLossV2 {

using namespace AscendC;

constexpr int64_t MAX_U_PAD = 256;
constexpr float FP32_MAX_V = 3.402823466e+38f;

__aicore__ inline float BitNegInf()
{
    union {
        uint32_t u;
        float f;
    } v;
    v.u = 0xFF800000u;
    return v.f;
}

__aicore__ inline float BitPosInf()
{
    union {
        uint32_t u;
        float f;
    } v;
    v.u = 0x7F800000u;
    return v.f;
}

__aicore__ inline bool IsNanFloat(float x)
{
    return !(x <= FP32_MAX_V) && !(x > FP32_MAX_V);
}

template <typename T>
class CtcLossV2 {
public:
    __aicore__ inline CtcLossV2(){};

    __aicore__ inline void Init(
        GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths,
        GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, const CtcLossV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline float SoftExp(float x);
    __aicore__ inline float SoftLn(float x);
    __aicore__ inline float LogAdd3(float l1, float l2, float l3);
    __aicore__ inline float LossFrom(float a, float b);
    __aicore__ inline float LpAt(int64_t t, int64_t n, int64_t c);
    __aicore__ inline int64_t ReadTarget(int64_t n, int64_t idx);
    __aicore__ inline int64_t ReadLen32(GlobalTensor<int32_t>& gm, int64_t n);
    __aicore__ inline int64_t ReadLen64(GlobalTensor<int64_t>& gm, int64_t n);
    __aicore__ inline void WriteAlphaRow(int64_t n, int64_t row, LocalTensor<float>& dst);
    __aicore__ inline void WriteLoss(int64_t n, float lossVal);
    __aicore__ inline void ProcessBatch(int64_t n);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> calcBuf_;
    TBuf<QuePosition::VECCALC> mathBuf_;
    TBuf<QuePosition::VECCALC> lpBuf_;
    TBuf<QuePosition::VECCALC> tgBuf_;

    GlobalTensor<T> logProbsGm_;
    GlobalTensor<int32_t> targetsGm32_;
    GlobalTensor<int64_t> targetsGm64_;
    GlobalTensor<int32_t> inputLenGm32_;
    GlobalTensor<int64_t> inputLenGm64_;
    GlobalTensor<int32_t> targetLenGm32_;
    GlobalTensor<int64_t> targetLenGm64_;
    GlobalTensor<T> lossGm_;
    GlobalTensor<T> alphaGm_;

    LocalTensor<float> lpLocal_;
    LocalTensor<int32_t> tgLocal32_;
    LocalTensor<int64_t> tgLocal64_;

    int64_t T_ = 0;
    int64_t N_ = 0;
    int64_t C_ = 0;
    int64_t S_ = 0;
    int64_t U_ = 0;
    int64_t UPad_ = 0;
    int64_t blank_ = 0;
    int64_t zeroInfinity_ = 0;
    int64_t targetsIsInt64_ = 0;
    int64_t usedCores_ = 1;
    int64_t batchBegin_ = 0;
    int64_t batchEnd_ = 0;
};

template <typename T>
__aicore__ inline float CtcLossV2<T>::SoftExp(float x)
{
    LocalTensor<float> tmp = mathBuf_.Get<float>();
    tmp.SetValue(0, x);
    Exp(tmp, tmp, 1);
    return tmp.GetValue(0);
}

template <typename T>
__aicore__ inline float CtcLossV2<T>::SoftLn(float x)
{
    LocalTensor<float> tmp = mathBuf_.Get<float>();
    tmp.SetValue(0, x);
    Ln(tmp, tmp, 1);
    return tmp.GetValue(0);
}

template <typename T>
__aicore__ inline float CtcLossV2<T>::LogAdd3(float l1, float l2, float l3)
{
    float lamax = l1;
    if (l2 > lamax) {
        lamax = l2;
    }
    if (l3 > lamax) {
        lamax = l3;
    }
    float sum = SoftExp(l1 - lamax) + SoftExp(l2 - lamax) + SoftExp(l3 - lamax);
    return SoftLn(sum) + lamax;
}

template <typename T>
__aicore__ inline float CtcLossV2<T>::LossFrom(float a, float b)
{
    const float negInf = BitNegInf();
    const float posInf = BitPosInf();
    float mx = 0.0f;
    float u0 = 0.0f;
    float u1 = 0.0f;
    if (a > b) {
        mx = a;
        u0 = 0.0f;
        u1 = b - a;
    } else if (a == negInf && b == negInf) {
        mx = 0.0f;
        u0 = a;
        u1 = b;
    } else {
        mx = b;
        u0 = a - b;
        u1 = 0.0f;
    }
    float val = -SoftLn(SoftExp(u0) + SoftExp(u1)) - mx;
    if ((val < posInf) || (zeroInfinity_ == 0) || IsNanFloat(val)) {
        return val;
    }
    return 0.0f;
}

template <typename T>
__aicore__ inline float CtcLossV2<T>::LpAt(int64_t t, int64_t n, int64_t c)
{
    // lpLocal_ 布局 [T, N, C] 展平
    return lpLocal_.GetValue((t * N_ + n) * C_ + c);
}

template <typename T>
__aicore__ inline int64_t CtcLossV2<T>::ReadTarget(int64_t n, int64_t idx)
{
    if (targetsIsInt64_ != 0) {
        return tgLocal64_.GetValue(n * S_ + idx);
    }
    return static_cast<int64_t>(tgLocal32_.GetValue(n * S_ + idx));
}

template <typename T>
__aicore__ inline int64_t CtcLossV2<T>::ReadLen32(GlobalTensor<int32_t>& gm, int64_t n)
{
    return static_cast<int64_t>(gm.GetValue(n));
}

template <typename T>
__aicore__ inline int64_t CtcLossV2<T>::ReadLen64(GlobalTensor<int64_t>& gm, int64_t n)
{
    return gm.GetValue(n);
}

template <typename T>
__aicore__ inline void CtcLossV2<T>::WriteAlphaRow(int64_t n, int64_t row, LocalTensor<float>& dst)
{
    LocalTensor<float> rowLocal = mathBuf_.Get<float>();
    for (int64_t s = 0; s < U_; ++s) {
        rowLocal.SetValue(s, dst.GetValue(s));
    }
    DataCopyExtParams p{1, static_cast<uint32_t>(U_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(alphaGm_[(n * T_ + row) * U_], rowLocal, p);
}

template <typename T>
__aicore__ inline void CtcLossV2<T>::WriteLoss(int64_t n, float lossVal)
{
    LocalTensor<float> outSlot = mathBuf_.Get<float>();
    outSlot.SetValue(0, lossVal);
    DataCopyExtParams p{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(lossGm_[n], outSlot, p);
}

template <typename T>
__aicore__ inline void CtcLossV2<T>::Init(
    GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths,
    GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, const CtcLossV2TilingData* tilingData)
{
    T_ = tilingData->timeSize;
    N_ = tilingData->batchSize;
    C_ = tilingData->classNum;
    S_ = tilingData->maxTargetLen;
    blank_ = tilingData->blank;
    zeroInfinity_ = tilingData->zeroInfinity;
    targetsIsInt64_ = tilingData->targetsIsInt64;
    usedCores_ = tilingData->usedCores;

    U_ = 2 * S_ + 1;
    UPad_ = ((U_ + 7) / 8) * 8;
    if (UPad_ > MAX_U_PAD) {
        UPad_ = MAX_U_PAD;
    }

    const int64_t blockIdx = GetBlockIdx();
    const int64_t perCore = (N_ + usedCores_ - 1) / usedCores_;
    batchBegin_ = blockIdx * perCore;
    batchEnd_ = batchBegin_ + perCore;
    if (batchBegin_ > N_) {
        batchBegin_ = N_;
    }
    if (batchEnd_ > N_) {
        batchEnd_ = N_;
    }

    const int64_t lpNum = T_ * N_ * C_;
    logProbsGm_.SetGlobalBuffer((__gm__ T*)log_probs, lpNum);
    if (targetsIsInt64_ != 0) {
        targetsGm64_.SetGlobalBuffer((__gm__ int64_t*)targets, N_ * S_);
        inputLenGm64_.SetGlobalBuffer((__gm__ int64_t*)input_lengths, N_);
        targetLenGm64_.SetGlobalBuffer((__gm__ int64_t*)target_lengths, N_);
    } else {
        targetsGm32_.SetGlobalBuffer((__gm__ int32_t*)targets, N_ * S_);
        inputLenGm32_.SetGlobalBuffer((__gm__ int32_t*)input_lengths, N_);
        targetLenGm32_.SetGlobalBuffer((__gm__ int32_t*)target_lengths, N_);
    }
    lossGm_.SetGlobalBuffer((__gm__ T*)neg_log_likelihood, N_);
    alphaGm_.SetGlobalBuffer((__gm__ T*)log_alpha, N_ * T_ * U_);

    pipe.InitBuffer(lpBuf_, lpNum * static_cast<int64_t>(sizeof(float)));
    if (targetsIsInt64_ != 0) {
        pipe.InitBuffer(tgBuf_, N_ * S_ * static_cast<int64_t>(sizeof(int64_t)));
    } else {
        pipe.InitBuffer(tgBuf_, N_ * S_ * static_cast<int64_t>(sizeof(int32_t)));
    }
    const int64_t calcBytes =
        2 * UPad_ * static_cast<int64_t>(sizeof(float)) + 2 * UPad_ * static_cast<int64_t>(sizeof(int32_t)) + 64;
    pipe.InitBuffer(calcBuf_, calcBytes);
    pipe.InitBuffer(mathBuf_, 1024);

    // 拷入 log_probs / targets，并 MTE2→Scalar 同步
    lpLocal_ = lpBuf_.Get<float>();
    {
        DataCopyExtParams p{1, static_cast<uint32_t>(lpNum * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(lpLocal_, logProbsGm_[0], p, pad);
    }
    if (targetsIsInt64_ != 0) {
        tgLocal64_ = tgBuf_.Get<int64_t>();
        DataCopyExtParams p{1, static_cast<uint32_t>(N_ * S_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPadExtParams<int64_t> pad{false, 0, 0, static_cast<int64_t>(0)};
        DataCopyPad(tgLocal64_, targetsGm64_[0], p, pad);
    } else {
        tgLocal32_ = tgBuf_.Get<int32_t>();
        DataCopyExtParams p{1, static_cast<uint32_t>(N_ * S_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(tgLocal32_, targetsGm32_[0], p, pad);
    }
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
}

template <typename T>
__aicore__ inline void CtcLossV2<T>::ProcessBatch(int64_t n)
{
    const float negInf = BitNegInf();
    const float posInf = BitPosInf();

    LocalTensor<float> dst = calcBuf_.Get<float>();
    LocalTensor<int32_t> intBuf = calcBuf_.GetWithOffset<int32_t>(
        static_cast<uint32_t>(2 * UPad_),
        static_cast<uint32_t>(2 * UPad_ * sizeof(float)));

    int64_t t_i = 0;
    int64_t s_i = 0;
    if (targetsIsInt64_ != 0) {
        t_i = ReadLen64(inputLenGm64_, n);
        s_i = ReadLen64(targetLenGm64_, n);
    } else {
        t_i = ReadLen32(inputLenGm32_, n);
        s_i = ReadLen32(targetLenGm32_, n);
    }

    if (t_i < s_i) {
        WriteLoss(n, (zeroInfinity_ != 0) ? 0.0f : posInf);
        return;
    }

    for (int64_t i = 0; i < UPad_; ++i) {
        intBuf.SetValue(i, 0);
        intBuf.SetValue(UPad_ + i, 0);
        dst.SetValue(i, negInf);
        dst.SetValue(UPad_ + i, negInf);
    }

    int64_t repeats = 0;
    int64_t idxCounter = 1;
    intBuf.SetValue(0, 1);
    if (s_i >= 1) {
        for (int64_t idx = 1; idx < s_i; ++idx) {
            const int64_t left = ReadTarget(n, idx - 1);
            const int64_t right = ReadTarget(n, idx);
            if (left == right) {
                intBuf.SetValue(idxCounter, 1);
                intBuf.SetValue(UPad_ + idxCounter - 1, 1);
                intBuf.SetValue(idxCounter + 1, 1);
                intBuf.SetValue(UPad_ + idxCounter, 1);
                idxCounter += 2;
                repeats += 1;
            } else {
                intBuf.SetValue(idxCounter, 2);
                intBuf.SetValue(UPad_ + idxCounter - 1, 2);
                idxCounter += 1;
            }
        }
    }
    intBuf.SetValue(UPad_ + idxCounter - 1, 1);

    dst.SetValue(0, LpAt(0, n, blank_));
    {
        const int64_t ct = ReadTarget(n, 0);
        dst.SetValue(1, (ct >= 0) ? LpAt(0, n, ct) : LpAt(0, n, blank_));
    }

    int64_t start = (repeats < t_i - s_i) ? 0 : 1;
    int64_t end = 2;

    for (int64_t t = 1; t < t_i; ++t) {
        for (int64_t i = 0; i < UPad_; ++i) {
            dst.SetValue(UPad_ + i, negInf);
        }
        const int64_t remain = s_i + repeats - t_i + t;
        if (remain >= 0) {
            start += static_cast<int64_t>(intBuf.GetValue(remain));
        }
        int64_t startLoop = start;
        if (t <= s_i + repeats) {
            end += static_cast<int64_t>(intBuf.GetValue(UPad_ + (t - 1)));
        }
        if (startLoop == 0) {
            dst.SetValue(UPad_ + 0, dst.GetValue(0) + LpAt(t, n, blank_));
            startLoop = 1;
        }
        for (int64_t s = startLoop; s < end; ++s) {
            int64_t ct = blank_;
            if ((s % 2) != 0) {
                ct = ReadTarget(n, s / 2);
            }
            const float clp = (ct >= 0) ? LpAt(t, n, ct) : LpAt(t, n, 0);
            const float l1 = dst.GetValue(s);
            const float l2 = dst.GetValue(s - 1);
            float l3 = negInf;
            if (((s % 2) != 0) && (s != 1)) {
                const int64_t nt = ReadTarget(n, s / 2 - 1);
                if (ct != nt) {
                    l3 = dst.GetValue(s - 2);
                }
            }
            dst.SetValue(UPad_ + s, LogAdd3(l1, l2, l3) + clp);
        }
        WriteAlphaRow(n, t - 1, dst);
        for (int64_t i = 0; i < UPad_; ++i) {
            dst.SetValue(i, dst.GetValue(UPad_ + i));
        }
    }
    WriteAlphaRow(n, t_i - 1, dst);

    const float a = dst.GetValue(2 * s_i);
    const float b = (s_i > 0) ? dst.GetValue(2 * s_i - 1) : negInf;
    WriteLoss(n, LossFrom(a, b));
}

template <typename T>
__aicore__ inline void CtcLossV2<T>::Process()
{
    // 先清零整份 log_alpha（对齐 reference 的 zeros）
    if (GetBlockIdx() == 0) {
        LocalTensor<float> z = mathBuf_.Get<float>();
        const int64_t total = N_ * T_ * U_;
        int64_t offset = 0;
        while (offset < total) {
            int64_t chunk = total - offset;
            if (chunk > 64) {
                chunk = 64;
            }
            for (int64_t i = 0; i < chunk; ++i) {
                z.SetValue(i, 0.0f);
            }
            DataCopyExtParams p{1, static_cast<uint32_t>(chunk * sizeof(float)), 0, 0, 0};
            DataCopyPad(alphaGm_[offset], z, p);
            offset += chunk;
        }
    }
    SyncAll();

    for (int64_t n = batchBegin_; n < batchEnd_; ++n) {
        ProcessBatch(n);
    }
}

} // namespace NsCtcLossV2
#endif // CTCLOSSV2_H
