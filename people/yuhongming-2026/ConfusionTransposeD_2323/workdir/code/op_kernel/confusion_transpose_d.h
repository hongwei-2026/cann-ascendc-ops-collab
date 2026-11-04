/*!
 * \file confusion_transpose_d.h
 * \brief ConfusionTransposeD 算子 kernel：permute+reshape 融合，按输出下标反查输入。
 */

#ifndef CONFUSIONTRANSPOSED_H
#define CONFUSIONTRANSPOSED_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "confusion_transpose_d_tiling_data.h"
#include "confusion_transpose_d_tiling_key.h"

namespace NsConfusionTransposeD {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t MAX_DIM = CTD_MAX_DIM;

template <typename T>
class ConfusionTransposeD {
public:
    __aicore__ inline ConfusionTransposeD(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ConfusionTransposeDTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadDims(const ConfusionTransposeDTilingData* td);
    __aicore__ inline int64_t MapSrcIndex(int64_t dstIndex) const;
    __aicore__ inline void CopyInFull();
    __aicore__ inline void ComputeTile(int64_t dstOffset, int64_t currentNum);
    __aicore__ inline void CopyOutTile(int64_t dstOffset, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockOffset_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubFactor_ = 0;
    int64_t transposeFirst_ = 1;
    int64_t rank_ = 0;
    int64_t rankIn_ = 0;

    int64_t dimIn_[MAX_DIM] = {0};
    int64_t dimShape_[MAX_DIM] = {0};
    int64_t perm_[MAX_DIM] = {0};
    int64_t invPerm_[MAX_DIM] = {0};
    int64_t tShape_[MAX_DIM] = {0};     // transpose 后中间 shape（TF=true）
    int64_t outShape_[MAX_DIM] = {0};   // 最终输出 shape
    int64_t strideIn_[MAX_DIM] = {0};
    int64_t strideT_[MAX_DIM] = {0};
    int64_t strideOut_[MAX_DIM] = {0};
    int64_t strideShape_[MAX_DIM] = {0};
};

template <typename T>
__aicore__ inline void ConfusionTransposeD<T>::LoadDims(const ConfusionTransposeDTilingData* td)
{
    dimIn_[0] = td->dimIn0; dimIn_[1] = td->dimIn1; dimIn_[2] = td->dimIn2; dimIn_[3] = td->dimIn3;
    dimIn_[4] = td->dimIn4; dimIn_[5] = td->dimIn5; dimIn_[6] = td->dimIn6; dimIn_[7] = td->dimIn7;
    dimShape_[0] = td->dimShape0; dimShape_[1] = td->dimShape1; dimShape_[2] = td->dimShape2; dimShape_[3] = td->dimShape3;
    dimShape_[4] = td->dimShape4; dimShape_[5] = td->dimShape5; dimShape_[6] = td->dimShape6; dimShape_[7] = td->dimShape7;
    perm_[0] = td->perm0; perm_[1] = td->perm1; perm_[2] = td->perm2; perm_[3] = td->perm3;
    perm_[4] = td->perm4; perm_[5] = td->perm5; perm_[6] = td->perm6; perm_[7] = td->perm7;

    for (int64_t i = 0; i < rank_; ++i) {
        invPerm_[perm_[i]] = i;
    }

    if (transposeFirst_ != 0) {
        for (int64_t i = 0; i < rank_; ++i) {
            tShape_[i] = dimIn_[perm_[i]];
            outShape_[i] = dimShape_[i];
        }
    } else {
        for (int64_t i = 0; i < rank_; ++i) {
            outShape_[i] = dimShape_[perm_[i]];
        }
    }

    // strides
    if (rankIn_ > 0) {
        strideIn_[rankIn_ - 1] = 1;
        for (int64_t i = rankIn_ - 2; i >= 0; --i) {
            strideIn_[i] = strideIn_[i + 1] * dimIn_[i + 1];
        }
    }
    if (rank_ > 0) {
        strideShape_[rank_ - 1] = 1;
        strideOut_[rank_ - 1] = 1;
        strideT_[rank_ - 1] = 1;
        for (int64_t i = rank_ - 2; i >= 0; --i) {
            strideShape_[i] = strideShape_[i + 1] * dimShape_[i + 1];
            strideOut_[i] = strideOut_[i + 1] * outShape_[i + 1];
            if (transposeFirst_ != 0) {
                strideT_[i] = strideT_[i + 1] * tShape_[i + 1];
            }
        }
    }
}

template <typename T>
__aicore__ inline int64_t ConfusionTransposeD<T>::MapSrcIndex(int64_t dstIndex) const
{
    if (transposeFirst_ != 0) {
        // dst flat == transposed contiguous flat → decode with tShape → map via perm → input
        int64_t rem = dstIndex;
        int64_t src = 0;
        for (int64_t i = 0; i < rank_; ++i) {
            int64_t coord = (strideT_[i] == 0) ? 0 : (rem / strideT_[i]);
            rem = (strideT_[i] == 0) ? 0 : (rem % strideT_[i]);
            // t_coord[i] comes from input dim perm[i]
            src += coord * strideIn_[perm_[i]];
        }
        return src;
    }
    // reshape first: decode dst with outShape → mid coords via invPerm on attr shape → flat
    int64_t rem = dstIndex;
    int64_t src = 0;
    int64_t midCoord[MAX_DIM];
    for (int64_t i = 0; i < rank_; ++i) {
        int64_t coord = (strideOut_[i] == 0) ? 0 : (rem / strideOut_[i]);
        rem = (strideOut_[i] == 0) ? 0 : (rem % strideOut_[i]);
        // out_coord[i] = mid[perm[i]] ⇒ mid[perm[i]] = out_coord[i]
        midCoord[perm_[i]] = coord;
    }
    for (int64_t i = 0; i < rank_; ++i) {
        src += midCoord[i] * strideShape_[i];
    }
    return src;
}

template <typename T>
__aicore__ inline void ConfusionTransposeD<T>::Init(GM_ADDR x, GM_ADDR y, const ConfusionTransposeDTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    ubFactor_ = tilingData->ubFactor;
    transposeFirst_ = tilingData->transposeFirst;
    rank_ = tilingData->rank;
    rankIn_ = tilingData->rankIn;

    const int64_t blockIdx = GetBlockIdx();
    blockOffset_ = blockIdx * tilingData->blockFactor;
    blockLength_ = tilingData->blockFactor;
    if (blockOffset_ >= totalNum_) {
        blockOffset_ = totalNum_;
        blockLength_ = 0;
    } else if (blockOffset_ + blockLength_ > totalNum_) {
        blockLength_ = totalNum_ - blockOffset_;
    }

    LoadDims(tilingData);

    inputGMX.SetGlobalBuffer((__gm__ T*)x, totalNum_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, totalNum_);

    // 输入整 tensor + 输出 tile
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, totalNum_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubFactor_ * sizeof(T));
}

template <typename T>
__aicore__ inline void ConfusionTransposeD<T>::CopyInFull()
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(totalNum_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[0], params, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void ConfusionTransposeD<T>::ComputeTile(int64_t dstOffset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    for (int64_t i = 0; i < currentNum; ++i) {
        const int64_t dstIndex = dstOffset + i;
        const int64_t srcIndex = MapSrcIndex(dstIndex);
        yLocal.SetValue(i, xLocal.GetValue(srcIndex));
    }
    outputQueueY.EnQue(yLocal);
    inputQueueX.EnQue(xLocal); // 归还，保持整 tensor 常驻下一 tile 复用
}

template <typename T>
__aicore__ inline void ConfusionTransposeD<T>::CopyOutTile(int64_t dstOffset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[dstOffset], yLocal, params);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void ConfusionTransposeD<T>::Process()
{
    if (blockLength_ <= 0 || totalNum_ <= 0) {
        return;
    }
    CopyInFull();
    // 预取：ComputeTile 会 DeQue 输入，需先保证已 EnQue
    int64_t remain = blockLength_;
    int64_t offset = blockOffset_;
    while (remain > 0) {
        const int64_t current = remain > ubFactor_ ? ubFactor_ : remain;
        ComputeTile(offset, current);
        CopyOutTile(offset, current);
        offset += current;
        remain -= current;
    }
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    inputQueueX.FreeTensor(xLocal);
}

} // namespace NsConfusionTransposeD
#endif // CONFUSIONTRANSPOSED_H
