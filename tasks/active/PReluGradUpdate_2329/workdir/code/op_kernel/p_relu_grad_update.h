/*!
 * \file p_relu_grad_update.h
 * \brief Kernel: Compare + Select; mode R/C (design_final)
 *
 * Select polarity (AscendC VSEL_TENSOR_TENSOR): bit1 -> src0, bit0 -> src1.
 * dx     = mask ? g : w*g   -> Select(dst, mask, g, wg)
 * update = mask ? 0 : x*g   -> Select(dst, mask, zeros, xg)
 * If polarity wrong on device, swap src0/src1 only (see EXP / B03).
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
    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx, GM_ADDR update,
        const PReluGradUpdateTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BuildWeightVecModeR();
    __aicore__ inline void FillWeightVecModeC(int64_t absOff, int64_t len, int64_t calcLen);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueG;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueDx;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueUpd;
    TBuf<TPosition::VECCALC> wBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> zeroBuf;

    GlobalTensor<T> gradsGm;
    GlobalTensor<T> featuresGm;
    GlobalTensor<T> weightsGm;
    GlobalTensor<T> dxGm;
    GlobalTensor<T> updateGm;

    int64_t blockStart_ = 0;
    int64_t blockLen_ = 0;
    int64_t tileLen_ = 0;
    int64_t channelNum_ = 0;
    int64_t innerSize_ = 0;
    int64_t sampleSize_ = 0;
    int64_t weightsPerChannel_ = 1;
    int64_t rowsPerTile_ = 0;
    T w0_{};
};

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::Init(
    GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx, GM_ADDR update,
    const PReluGradUpdateTilingData* tilingData)
{
    gradsGm.SetGlobalBuffer((__gm__ T*)grads);
    featuresGm.SetGlobalBuffer((__gm__ T*)features);
    weightsGm.SetGlobalBuffer((__gm__ T*)weights);
    dxGm.SetGlobalBuffer((__gm__ T*)dx);
    updateGm.SetGlobalBuffer((__gm__ T*)update);

    const int64_t M = tilingData->totalNum;
    const int64_t blockFactor = tilingData->blockFactor;
    blockStart_ = GetBlockIdx() * blockFactor;
    blockLen_ = 0;
    if (blockStart_ < M) {
        blockLen_ = blockFactor;
        if (blockStart_ + blockLen_ > M) {
            blockLen_ = M - blockStart_;
        }
    }

    tileLen_ = tilingData->tileLen;
    channelNum_ = tilingData->channelNum;
    innerSize_ = tilingData->innerSize;
    sampleSize_ = tilingData->sampleSize;
    weightsPerChannel_ = tilingData->weightsPerChannel;
    rowsPerTile_ = tilingData->rowsPerTile;
    w0_ = weightsGm.GetValue(0);

    if (blockLen_ <= 0 || tileLen_ <= 0) {
        return;
    }

    const uint32_t tBytes = static_cast<uint32_t>(sizeof(T));
    pipe.InitBuffer(inQueueG, BUFFER_NUM, tileLen_ * tBytes);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLen_ * tBytes);
    pipe.InitBuffer(outQueueDx, BUFFER_NUM, tileLen_ * tBytes);
    pipe.InitBuffer(outQueueUpd, BUFFER_NUM, tileLen_ * tBytes);
    pipe.InitBuffer(wBuf, tileLen_ * sizeof(float));
    pipe.InitBuffer(maskBuf, tileLen_ * sizeof(uint8_t));
    pipe.InitBuffer(tmpBuf, tileLen_ * sizeof(float) * 4);
    pipe.InitBuffer(zeroBuf, tileLen_ * sizeof(float));

    Duplicate(zeroBuf.Get<float>(), 0.0f, tileLen_);
    if (weightsPerChannel_ == 0) {
        Duplicate(wBuf.Get<float>(), static_cast<float>(w0_), tileLen_);
    } else if (rowsPerTile_ > 0) {
        BuildWeightVecModeR();
    }
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::BuildWeightVecModeR()
{
    LocalTensor<float> wVec = wBuf.Get<float>();
    for (int64_t r = 0; r < rowsPerTile_; ++r) {
        for (int64_t ch = 0; ch < channelNum_; ++ch) {
            const float wv = static_cast<float>(weightsGm.GetValue(ch));
            const int64_t off = r * sampleSize_ + ch * innerSize_;
            Duplicate(wVec[off], wv, innerSize_);
        }
    }
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::FillWeightVecModeC(int64_t absOff, int64_t len, int64_t calcLen)
{
    LocalTensor<float> wVec = wBuf.Get<float>();
    for (int64_t i = 0; i < len; ++i) {
        const int64_t idx = absOff + i;
        int64_t c = 0;
        if (innerSize_ > 0 && channelNum_ > 0) {
            c = (idx / innerSize_) % channelNum_;
        }
        wVec.SetValue(i, static_cast<float>(weightsGm.GetValue(c)));
    }
    for (int64_t i = len; i < calcLen; ++i) {
        wVec.SetValue(i, 0.0f);
    }
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::Process()
{
    if (blockLen_ <= 0 || tileLen_ <= 0) {
        return;
    }
    const int64_t alignElem = static_cast<int64_t>(32 / sizeof(T));
    int64_t done = 0;
    while (done < blockLen_) {
        int64_t len = tileLen_;
        if (done + len > blockLen_) {
            len = blockLen_ - done;
        }
        int64_t calcLen = ((len + alignElem - 1) / alignElem) * alignElem;
        if (calcLen > tileLen_) {
            calcLen = tileLen_;
            len = tileLen_;
        }
        const int64_t absOff = blockStart_ + done;

        if (rowsPerTile_ == 0 && weightsPerChannel_ == 1) {
            FillWeightVecModeC(absOff, len, calcLen);
        } else if (rowsPerTile_ == 0 && weightsPerChannel_ == 0) {
            Duplicate(wBuf.Get<float>(), static_cast<float>(w0_), calcLen);
        }

        LocalTensor<T> gLocal = inQueueG.AllocTensor<T>();
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(gLocal, gradsGm[absOff], calcLen);
        DataCopy(xLocal, featuresGm[absOff], calcLen);
        inQueueG.EnQue(gLocal);
        inQueueX.EnQue(xLocal);
        gLocal = inQueueG.DeQue<T>();
        xLocal = inQueueX.DeQue<T>();

        LocalTensor<T> dxLocal = outQueueDx.AllocTensor<T>();
        LocalTensor<T> upLocal = outQueueUpd.AllocTensor<T>();

        LocalTensor<float> tmp = tmpBuf.Get<float>();
        LocalTensor<float> gF = tmp;
        LocalTensor<float> xF = tmp[tileLen_];
        LocalTensor<float> stashG = tmp[tileLen_ * 2];
        LocalTensor<float> xg = tmp[tileLen_ * 3];
        LocalTensor<float> wVec = wBuf.Get<float>();
        LocalTensor<float> zeros = zeroBuf.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();

        Cast(gF, gLocal, RoundMode::CAST_NONE, calcLen);
        Cast(xF, xLocal, RoundMode::CAST_NONE, calcLen);
        DataCopy(stashG, gF, calcLen);

        Compare(mask, xF, zeros, CMPMODE::GT, calcLen);
        Mul(xg, wVec, stashG, calcLen); // w * g
        Select(gF, mask, stashG, xg, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcLen); // dx
        Mul(xg, xF, stashG, calcLen);                                             // x * g
        Select(stashG, mask, zeros, xg, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcLen); // update

        Cast(dxLocal, gF, RoundMode::CAST_RINT, calcLen);
        Cast(upLocal, stashG, RoundMode::CAST_RINT, calcLen);

        outQueueDx.EnQue(dxLocal);
        outQueueUpd.EnQue(upLocal);
        dxLocal = outQueueDx.DeQue<T>();
        upLocal = outQueueUpd.DeQue<T>();

        // write only real elements (avoid padding overwrite past M)
        if (len == calcLen) {
            DataCopy(dxGm[absOff], dxLocal, calcLen);
            DataCopy(updateGm[absOff], upLocal, calcLen);
        } else {
            for (int64_t i = 0; i < len; ++i) {
                dxGm.SetValue(absOff + i, dxLocal.GetValue(i));
                updateGm.SetValue(absOff + i, upLocal.GetValue(i));
            }
        }

        outQueueDx.FreeTensor(dxLocal);
        outQueueUpd.FreeTensor(upLocal);
        inQueueG.FreeTensor(gLocal);
        inQueueX.FreeTensor(xLocal);

        done += len;
    }
}

} // namespace NsPReluGradUpdate
#endif // PRELUGRADUPDATE_H
