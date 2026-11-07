/*!
 * \file pad_v3_grad.h
 * \brief PadV3Grad：把 Pad 之后的梯度折回原张量。
 *
 * constant 只保留中间那一块，按输出行分核拷贝。
 * edge / reflect / symmetric / circular 按维把多个来源加到同一个输出位置。
 * 累加在 float32。非 constant 用单核和两块 float workspace，不调用 SyncAll。
 * 下标与 samples/reference.py 的 _pad_index 一致，负数取模跟 numpy 一样补成正数。
 */

#ifndef PADV3GRAD_H
#define PADV3GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pad_v3_grad_tiling_data.h"
#include "pad_v3_grad_tiling_key.h"

namespace NsPadV3Grad {

using namespace AscendC;

constexpr int64_t kModeConstant = 0;
constexpr int64_t kModeEdge = 1;
constexpr int64_t kModeReflect = 2;
constexpr int64_t kModeSymmetric = 3;
constexpr int64_t kAccFloats = 8192;
constexpr int64_t kTile = 256;

template <typename T>
class PadV3Grad {
public:
    __aicore__ inline PadV3Grad() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace,
        const PadV3GradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t PyMod(int64_t a, int64_t m) const;
    __aicore__ inline int64_t MapCoord(int64_t coord, int64_t front, int64_t n) const;
    __aicore__ inline int64_t Align64(int64_t n) const;
    __aicore__ inline void LoadX(int64_t offset, const LocalTensor<float>& dst, int64_t valid, int64_t align);
    __aicore__ inline void StoreY(int64_t offset, const LocalTensor<float>& src, int64_t valid);
    __aicore__ inline void LoadFloat(const GlobalTensor<float>& gm, int64_t offset, const LocalTensor<float>& dst,
        int64_t valid, int64_t align);
    __aicore__ inline void StoreFloat(const GlobalTensor<float>& gm, int64_t offset, const LocalTensor<float>& src,
        int64_t valid);
    __aicore__ inline void CopyXToWorkspace();
    __aicore__ inline void StoreWorkspaceToY(const GlobalTensor<float>& src);
    __aicore__ inline void FoldAxis(GlobalTensor<float>& src, GlobalTensor<float>& dst, int64_t axis);
    __aicore__ inline int64_t InteriorSrc(int64_t row) const;
    __aicore__ inline void CopyConstantRow(int64_t row);
    __aicore__ inline void ProcessConstant();
    __aicore__ inline void ProcessFold();

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> acc_;
    TBuf<TPosition::VECCALC> src_;
    TBuf<TPosition::VECCALC> raw_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> ws0_;
    GlobalTensor<float> ws1_;

    int64_t rank_ = 1;
    int64_t mode_ = kModeReflect;
    int64_t totalX_ = 0;
    int64_t totalY_ = 0;
    int64_t usedCores_ = 1;
    int64_t rowsPerCore_ = 0;
    int64_t rowRemain_ = 0;
    int64_t yRowLength_ = 0;
    int64_t xDim_[8] = {};
    int64_t yDim_[8] = {};
    int64_t front_[8] = {};
    int64_t back_[8] = {};
    int64_t curDim_[8] = {};
};

template <typename T>
__aicore__ inline int64_t PadV3Grad<T>::PyMod(int64_t a, int64_t m) const
{
    if (m <= 0) {
        return 0;
    }
    int64_t r = a % m;
    if (r < 0) {
        r += m;
    }
    return r;
}

template <typename T>
__aicore__ inline int64_t PadV3Grad<T>::MapCoord(int64_t coord, int64_t front, int64_t n) const
{
    const int64_t j = coord - front;
    if (mode_ == kModeEdge) {
        if (n <= 1) {
            return 0;
        }
        if (j < 0) {
            return 0;
        }
        if (j >= n) {
            return n - 1;
        }
        return j;
    }
    if (n <= 1) {
        return 0;
    }
    if (mode_ == kModeReflect) {
        const int64_t period = 2 * (n - 1);
        const int64_t k = PyMod(j, period);
        return k < n ? k : period - k;
    }
    if (mode_ == kModeSymmetric) {
        const int64_t period = 2 * n;
        const int64_t k = PyMod(j, period);
        return k < n ? k : period - 1 - k;
    }
    return PyMod(j, n);
}

template <typename T>
__aicore__ inline int64_t PadV3Grad<T>::Align64(int64_t n) const
{
    return (n + 63) & ~static_cast<int64_t>(63);
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace,
    const PadV3GradTilingData* tilingData)
{
    (void)paddings;
    rank_ = tilingData->rank > 0 ? tilingData->rank : 1;
    mode_ = tilingData->mode;
    totalX_ = tilingData->totalX;
    totalY_ = tilingData->totalY;
    usedCores_ = tilingData->usedCores > 0 ? tilingData->usedCores : 1;
    rowsPerCore_ = tilingData->rowsPerCore;
    rowRemain_ = tilingData->rowRemain;
    yRowLength_ = tilingData->yRowLength;
    for (int64_t i = 0; i < 8; ++i) {
        xDim_[i] = tilingData->xDim[i];
        yDim_[i] = tilingData->yDim[i];
        front_[i] = tilingData->front[i];
        back_[i] = tilingData->back[i];
        curDim_[i] = xDim_[i];
    }
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalX_);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalY_ > 0 ? totalY_ : 1);
    if (workspace != nullptr && totalX_ > 0) {
        const int64_t bytes = (totalX_ * static_cast<int64_t>(sizeof(float)) + 31) & ~static_cast<int64_t>(31);
        const int64_t gap = bytes / static_cast<int64_t>(sizeof(float));
        ws0_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), totalX_);
        ws1_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + gap, totalX_);
    }
    pipe_.InitBuffer(acc_, kAccFloats * static_cast<int64_t>(sizeof(float)));
    pipe_.InitBuffer(src_, kTile * static_cast<int64_t>(sizeof(float)));
    pipe_.InitBuffer(raw_, kTile * static_cast<int64_t>(sizeof(float)));
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::LoadX(int64_t offset, const LocalTensor<float>& dst, int64_t valid, int64_t align)
{
    Duplicate(dst, 0.0f, static_cast<uint32_t>(align));
    PipeBarrier<PIPE_ALL>();
    if (valid <= 0) {
        return;
    }
    if constexpr (sizeof(T) == sizeof(float)) {
        DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(dst, xGm_[offset], p, pad);
    } else {
        LocalTensor<T> raw = raw_.Get<T>();
        DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(raw, xGm_[offset], p, pad);
        PipeBarrier<PIPE_ALL>();
        Cast(dst, raw, RoundMode::CAST_NONE, static_cast<uint32_t>(align));
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::StoreY(int64_t offset, const LocalTensor<float>& src, int64_t valid)
{
    if (valid <= 0) {
        return;
    }
    if constexpr (sizeof(T) == sizeof(float)) {
        DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
        DataCopyPad(yGm_[offset], src, p);
    } else {
        LocalTensor<T> raw = raw_.Get<T>();
        Cast(raw, src, RoundMode::CAST_RINT, static_cast<uint32_t>(Align64(valid)));
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(yGm_[offset], raw, p);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::LoadFloat(const GlobalTensor<float>& gm, int64_t offset,
    const LocalTensor<float>& dst, int64_t valid, int64_t align)
{
    Duplicate(dst, 0.0f, static_cast<uint32_t>(align));
    PipeBarrier<PIPE_ALL>();
    if (valid <= 0) {
        return;
    }
    DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyPad(dst, gm[offset], p, pad);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::StoreFloat(const GlobalTensor<float>& gm, int64_t offset,
    const LocalTensor<float>& src, int64_t valid)
{
    if (valid <= 0) {
        return;
    }
    DataCopyExtParams p{1, static_cast<uint32_t>(valid * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
    DataCopyPad(gm[offset], src, p);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::CopyXToWorkspace()
{
    LocalTensor<float> tile = src_.Get<float>();
    for (int64_t off = 0; off < totalX_; off += kTile) {
        const int64_t valid = totalX_ - off > kTile ? kTile : totalX_ - off;
        LoadX(off, tile, valid, Align64(valid));
        StoreFloat(ws0_, off, tile, valid);
    }
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::StoreWorkspaceToY(const GlobalTensor<float>& src)
{
    LocalTensor<float> tile = src_.Get<float>();
    for (int64_t off = 0; off < totalY_; off += kTile) {
        const int64_t valid = totalY_ - off > kTile ? kTile : totalY_ - off;
        LoadFloat(src, off, tile, valid, Align64(valid));
        StoreY(off, tile, valid);
    }
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::FoldAxis(GlobalTensor<float>& srcGm, GlobalTensor<float>& dstGm, int64_t axis)
{
    const int64_t extent = curDim_[axis];
    const int64_t n = yDim_[axis];
    int64_t inner = 1;
    int64_t outer = 1;
    for (int64_t i = axis + 1; i < rank_; ++i) {
        inner *= curDim_[i];
    }
    for (int64_t i = 0; i < axis; ++i) {
        outer *= curDim_[i];
    }
    if (extent <= 0 || n <= 0 || inner <= 0 || outer <= 0) {
        return;
    }
    const int64_t tile = inner < kTile ? inner : kTile;
    const int64_t align = Align64(tile);
    int64_t chunk = align > 0 ? kAccFloats / align : 1;
    if (chunk < 1) {
        chunk = 1;
    }
    LocalTensor<float> acc = acc_.Get<float>();
    LocalTensor<float> tileBuf = src_.Get<float>();
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t dst0 = 0; dst0 < n; dst0 += chunk) {
            const int64_t chunkNow = n - dst0 > chunk ? chunk : n - dst0;
            for (int64_t in0 = 0; in0 < inner; in0 += tile) {
                const int64_t tileNow = inner - in0 > tile ? tile : inner - in0;
                Duplicate(acc, 0.0f, static_cast<uint32_t>(chunkNow * align));
                PipeBarrier<PIPE_ALL>();
                for (int64_t s = 0; s < extent; ++s) {
                    const int64_t mapped = MapCoord(s, front_[axis], n);
                    if (mapped < dst0 || mapped >= dst0 + chunkNow) {
                        continue;
                    }
                    const int64_t local = mapped - dst0;
                    const int64_t srcOff = (o * extent + s) * inner + in0;
                    LoadFloat(srcGm, srcOff, tileBuf, tileNow, align);
                    Add(acc[static_cast<uint32_t>(local * align)], acc[static_cast<uint32_t>(local * align)], tileBuf,
                        static_cast<uint32_t>(align));
                    PipeBarrier<PIPE_ALL>();
                }
                for (int64_t local = 0; local < chunkNow; ++local) {
                    const int64_t dstOff = (o * n + dst0 + local) * inner + in0;
                    StoreFloat(dstGm, dstOff, acc[static_cast<uint32_t>(local * align)], tileNow);
                }
            }
        }
    }
    curDim_[axis] = n;
}

template <typename T>
__aicore__ inline int64_t PadV3Grad<T>::InteriorSrc(int64_t row) const
{
    int64_t coord[8] = {};
    int64_t tmp = row;
    for (int64_t d = rank_ - 2; d >= 0; --d) {
        const int64_t dim = yDim_[d] > 0 ? yDim_[d] : 1;
        coord[d] = tmp % dim;
        tmp /= dim;
    }
    int64_t xStride[8] = {};
    int64_t stride = 1;
    for (int64_t d = rank_ - 1; d >= 0; --d) {
        xStride[d] = stride;
        stride *= xDim_[d] > 0 ? xDim_[d] : 1;
    }
    int64_t src = 0;
    for (int64_t d = 0; d < rank_ - 1; ++d) {
        src += (coord[d] + front_[d]) * xStride[d];
    }
    src += front_[rank_ - 1] * xStride[rank_ - 1];
    return src;
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::CopyConstantRow(int64_t row)
{
    const int64_t srcBase = InteriorSrc(row);
    const int64_t dstBase = row * yRowLength_;
    LocalTensor<float> tile = src_.Get<float>();
    for (int64_t off = 0; off < yRowLength_; off += kTile) {
        const int64_t valid = yRowLength_ - off > kTile ? kTile : yRowLength_ - off;
        LoadX(srcBase + off, tile, valid, Align64(valid));
        StoreY(dstBase + off, tile, valid);
    }
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::ProcessConstant()
{
    const int64_t core = static_cast<int64_t>(GetBlockIdx());
    if (core >= usedCores_ || yRowLength_ <= 0 || totalY_ <= 0) {
        return;
    }
    const int64_t begin = core * rowsPerCore_ + (core < rowRemain_ ? core : rowRemain_);
    const int64_t count = rowsPerCore_ + (core < rowRemain_ ? 1 : 0);
    for (int64_t row = begin; row < begin + count; ++row) {
        CopyConstantRow(row);
    }
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::ProcessFold()
{
    if (static_cast<int64_t>(GetBlockIdx()) != 0 || totalY_ <= 0 || totalX_ <= 0) {
        return;
    }
    CopyXToWorkspace();
    GlobalTensor<float> cur = ws0_;
    GlobalTensor<float> other = ws1_;
    for (int64_t axis = 0; axis < rank_; ++axis) {
        if (front_[axis] == 0 && back_[axis] == 0) {
            continue;
        }
        FoldAxis(cur, other, axis);
        GlobalTensor<float> swap = cur;
        cur = other;
        other = swap;
    }
    StoreWorkspaceToY(cur);
}

template <typename T>
__aicore__ inline void PadV3Grad<T>::Process()
{
    if (mode_ == kModeConstant) {
        ProcessConstant();
        return;
    }
    ProcessFold();
}

} // namespace NsPadV3Grad
#endif
