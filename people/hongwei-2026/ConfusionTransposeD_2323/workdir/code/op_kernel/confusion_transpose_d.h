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
    __aicore__ inline void CopyRun(int64_t dstOffset, int64_t srcOffset, int64_t count);
    __aicore__ inline void ProcessCopyRows();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockOffset_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubFactor_ = 0;
    int64_t transposeFirst_ = 1;
    int64_t rank_ = 0;
    int64_t rankIn_ = 0;
    int64_t copySpan_ = 1;

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
    copySpan_ = tilingData->copySpan;

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

    if (copySpan_ > 1) {
        // MTE2→MTE3 直通，不分配整输入，也不经过标量/向量寄存器。
        pipe.InitBuffer(copyQueue, BUFFER_NUM, ubFactor_ * sizeof(T));
    } else {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, totalNum_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubFactor_ * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void ConfusionTransposeD<T>::CopyRun(int64_t dstOffset, int64_t srcOffset, int64_t count)
{
    LocalTensor<T> local = copyQueue.AllocTensor<T>();
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(local, inputGMX[srcOffset], params, pad);
    copyQueue.EnQue(local);
    local = copyQueue.DeQue<T>();
    DataCopyPad(outputGMY[dstOffset], local, params);
    copyQueue.FreeTensor(local);
}

template <typename T>
__aicore__ inline void ConfusionTransposeD<T>::ProcessCopyRows()
{
    const int64_t align = 32 / sizeof(T);
    const int64_t paddedRow = ((copySpan_ + align - 1) / align) * align;
    const int64_t end = blockOffset_ + blockLength_;
    if (paddedRow > ubFactor_) {
        // 连续行超过输出 tile 时，只切该行；不扩大超 UB tensor 支持范围。
        for (int64_t row = blockOffset_; row < end; row += copySpan_) {
            const int64_t src = MapSrcIndex(row);
            for (int64_t start = 0; start < copySpan_; start += ubFactor_) {
                const int64_t count = (copySpan_ - start > ubFactor_) ? ubFactor_ : copySpan_ - start;
                CopyRun(row + start, src + start, count);
            }
        }
        return;
    }
    int64_t rowsPerTile = ubFactor_ / paddedRow;
    if (rowsPerTile > 4095) {
        rowsPerTile = 4095;
    }
    for (int64_t dst = blockOffset_; dst < end;) {
        const int64_t remainingRows = (end - dst) / copySpan_;
        const int64_t maxRows = remainingRows > rowsPerTile ? rowsPerTile : remainingRows;
        int64_t rows = maxRows;
        const int64_t rowBytes = copySpan_ * static_cast<int64_t>(sizeof(T));
        const int64_t src = MapSrcIndex(dst);
        int64_t srcStep = 0;
        int64_t batchRows = 1;
        bool canBatch = rowBytes % 32 == 0 && maxRows > 1;
        if (canBatch) {
            srcStep = MapSrcIndex(dst + copySpan_) - src;
            canBatch = srcStep >= copySpan_;
            canBatch = canBatch && (srcStep - copySpan_) * static_cast<int64_t>(sizeof(T)) <= UINT32_MAX;
            if (canBatch) {
                batchRows = 2;
                for (int64_t row = 2; row < maxRows; ++row) {
                    if (MapSrcIndex(dst + row * copySpan_) != src + row * srcStep) {
                        break;
                    }
                    ++batchRows;
                }
            }
        }
        if (canBatch && batchRows > 1) {
            rows = batchRows;
            // 等差源行合并成一次MTE2块DMA；行字节32B对齐时UB无需行间补齐。
            LocalTensor<T> local = copyQueue.AllocTensor<T>();
            DataCopyExtParams inParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(rowBytes),
                static_cast<uint32_t>((srcStep - copySpan_) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(local, inputGMX[src], inParams, pad);
            copyQueue.EnQue(local);
            local = copyQueue.DeQue<T>();
            DataCopyExtParams outParams{1, static_cast<uint32_t>(rows * rowBytes), 0, 0, 0};
            DataCopyPad(outputGMY[dst], local, outParams);
            copyQueue.FreeTensor(local);
            dst += rows * copySpan_;
            continue;
        }
        LocalTensor<T> local = copyQueue.AllocTensor<T>();
        DataCopyExtParams inParams{1, static_cast<uint32_t>(copySpan_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        for (int64_t row = 0; row < rows; ++row) {
            const int64_t src = MapSrcIndex(dst + row * copySpan_);
            DataCopyPad(local[row * paddedRow], inputGMX[src], inParams, pad);
        }
        copyQueue.EnQue(local);
        local = copyQueue.DeQue<T>();
        // 每个 UB 行自动跳过 32B 尾部补齐，GM 行紧密相连。
        DataCopyExtParams outParams{static_cast<uint16_t>(rows),
            static_cast<uint32_t>(copySpan_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[dst], local, outParams);
        copyQueue.FreeTensor(local);
        dst += rows * copySpan_;
    }
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
    if (copySpan_ > 1) {
        ProcessCopyRows();
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
