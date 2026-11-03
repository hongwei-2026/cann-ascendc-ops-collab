/*!
 * \file multilabel_margin_loss.h
 * \brief MultilabelMarginLoss AscendC kernel（对齐 910B TBE 计数语义）
 *
 * is_target：扫描到首个 <0 为止，对合法类标计数（int32）。
 * loss：对非目标类 j 与目标类 l，累加 cnt[l]*max(0, 1+x[j]-x[l]) 再 /C。
 * 中间一律 float32；fp16 Cast，bf16 位转换（避免 910B bf16 Cast 缺指令）。
 */

#ifndef MULTILABELMARGINLOSS_H
#define MULTILABELMARGINLOSS_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "multilabel_margin_loss_tiling_data.h"
#include "multilabel_margin_loss_tiling_key.h"

namespace NsMultilabelMarginLoss {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t REDUCTION_NONE = 0;
constexpr int64_t REDUCTION_MEAN = 1;
constexpr int64_t REDUCTION_SUM = 2;

__aicore__ inline float Bf16ToFloat(bfloat16_t v)
{
    uint16_t bits = *reinterpret_cast<uint16_t*>(&v);
    float out = 0.0f;
    *reinterpret_cast<uint32_t*>(&out) = static_cast<uint32_t>(bits) << 16;
    return out;
}

__aicore__ inline bfloat16_t FloatToBf16Rne(float f)
{
    uint32_t u = *reinterpret_cast<uint32_t*>(&f);
    uint32_t lsb = (u >> 16) & 1U;
    uint16_t bits = static_cast<uint16_t>((u + 0x7FFFU + lsb) >> 16);
    bfloat16_t out;
    *reinterpret_cast<uint16_t*>(&out) = bits;
    return out;
}

template <typename T>
class MultilabelMarginLoss {
public:
    __aicore__ inline MultilabelMarginLoss() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR y, GM_ADDR isTarget,
        GM_ADDR workspace, const MultilabelMarginLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SyncSV();
    __aicore__ inline void SyncSMTE3();
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment) const;
    __aicore__ inline float LoadXAsFloat(const LocalTensor<T>& xLocal, int64_t idx);
    __aicore__ inline void StoreYValue(int64_t idx, float value);
    __aicore__ inline float ProcessOneRow(int64_t row);

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> targetBuf_;
    TBuf<TPosition::VECCALC> isTargetBuf_;
    TBuf<TPosition::VECCALC> xFloatBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> targetGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> isTargetGm_;
    GlobalTensor<float> rowLossGm_;

    int64_t batchSize_ = 0;
    int64_t classNum_ = 0;
    int64_t targetLen_ = 0;
    int64_t reduction_ = 1;
    int64_t usedCores_ = 1;
    int64_t rowsPerCore_ = 1;
    int64_t classTile_ = 0;
    int64_t targetTile_ = 0;
    int64_t rowStart_ = 0;
    int64_t rowCount_ = 0;
};

template <typename T>
__aicore__ inline int64_t MultilabelMarginLoss<T>::AlignUp(int64_t value, int64_t alignment) const
{
    return ((value + alignment - 1) / alignment) * alignment;
}

template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::SyncSV()
{
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(evt);
    WaitFlag<HardEvent::S_V>(evt);
}

template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::SyncSMTE3()
{
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evt);
    WaitFlag<HardEvent::S_MTE3>(evt);
}

template <typename T>
__aicore__ inline float MultilabelMarginLoss<T>::LoadXAsFloat(const LocalTensor<T>& xLocal, int64_t idx)
{
    T raw = xLocal.GetValue(idx);
    if constexpr (std::is_same<T, float>::value) {
        return raw;
    } else if constexpr (std::is_same<T, half>::value) {
        // 标量路径：half 按 IEEE 位展开到 float（与 Cast CAST_NONE 一致量级）
        uint16_t h = *reinterpret_cast<uint16_t*>(&raw);
        uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
        uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
        uint32_t man = static_cast<uint32_t>(h) & 0x3FFu;
        uint32_t outBits;
        if (exp == 0) {
            if (man == 0) {
                outBits = sign;
            } else {
                exp = 127 - 15 + 1;
                while ((man & 0x400u) == 0) {
                    man <<= 1;
                    --exp;
                }
                man &= 0x3FFu;
                outBits = sign | (exp << 23) | (man << 13);
            }
        } else if (exp == 31) {
            outBits = sign | 0x7F800000u | (man << 13);
        } else {
            outBits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
        }
        float out = 0.0f;
        *reinterpret_cast<uint32_t*>(&out) = outBits;
        return out;
    } else {
        return Bf16ToFloat(raw);
    }
}

template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::StoreYValue(int64_t idx, float value)
{
    T out;
    if constexpr (std::is_same<T, float>::value) {
        out = value;
    } else if constexpr (std::is_same<T, half>::value) {
        // float → half RNE
        uint32_t f = *reinterpret_cast<uint32_t*>(&value);
        uint32_t sign = (f >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((f >> 23) & 0xFFu) - 127 + 15;
        uint32_t man = f & 0x7FFFFFu;
        uint16_t h;
        if ((f & 0x7F800000u) == 0x7F800000u) {
            h = static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
        } else if (exp <= 0) {
            h = static_cast<uint16_t>(sign);
        } else if (exp >= 31) {
            h = static_cast<uint16_t>(sign | 0x7C00u);
        } else {
            uint32_t rounded = man + 0x1000u;
            h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (rounded >> 13));
        }
        *reinterpret_cast<uint16_t*>(&out) = h;
    } else {
        out = FloatToBf16Rne(value);
    }
    yGm_.SetValue(idx, out);
}

template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::Init(GM_ADDR x, GM_ADDR target, GM_ADDR y, GM_ADDR isTarget,
    GM_ADDR workspace, const MultilabelMarginLossTilingData* tilingData)
{
    batchSize_ = tilingData->batchSize;
    classNum_ = tilingData->classNum;
    targetLen_ = tilingData->targetLen;
    reduction_ = tilingData->reduction;
    usedCores_ = tilingData->usedCores;
    rowsPerCore_ = tilingData->rowsPerCore;
    classTile_ = tilingData->classTile > 0 ? tilingData->classTile : tilingData->classNum;
    targetTile_ = tilingData->targetTile > 0 ? tilingData->targetTile : tilingData->targetLen;
    if (classTile_ > classNum_) {
        classTile_ = classNum_;
    }
    if (targetTile_ > targetLen_) {
        targetTile_ = targetLen_;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    rowStart_ = blockIdx * rowsPerCore_;
    rowCount_ = 0;
    if (rowStart_ < batchSize_) {
        rowCount_ = rowsPerCore_;
        if (rowStart_ + rowCount_ > batchSize_) {
            rowCount_ = batchSize_ - rowStart_;
        }
    }

    xGm_.SetGlobalBuffer((__gm__ T*)x, batchSize_ * classNum_);
    targetGm_.SetGlobalBuffer((__gm__ int32_t*)target, batchSize_ * targetLen_);
    if (reduction_ == REDUCTION_NONE) {
        yGm_.SetGlobalBuffer((__gm__ T*)y, batchSize_);
    } else {
        yGm_.SetGlobalBuffer((__gm__ T*)y, 1);
    }
    isTargetGm_.SetGlobalBuffer((__gm__ int32_t*)isTarget, batchSize_ * classNum_);

    GM_ADDR userWs = GetUserWorkspace(workspace);
    rowLossGm_.SetGlobalBuffer((__gm__ float*)userWs, batchSize_);

    // UB 按 classTile/targetTile 分配（大 C 滑窗，对齐 design_final）
    const int64_t xAlign = AlignUp(classTile_, 32 / static_cast<int64_t>(sizeof(T)));
    const int64_t tAlign = AlignUp(targetTile_, 8);
    const int64_t cAlign = AlignUp(classTile_, 8);

    pipe_.InitBuffer(xBuf_, xAlign * sizeof(T));
    pipe_.InitBuffer(targetBuf_, tAlign * sizeof(int32_t));
    pipe_.InitBuffer(isTargetBuf_, cAlign * sizeof(int32_t));
    pipe_.InitBuffer(xFloatBuf_, AlignUp(classTile_, 8) * sizeof(float));
}

template <typename T>
__aicore__ inline float MultilabelMarginLoss<T>::ProcessOneRow(int64_t row)
{
    LocalTensor<T> xLocal = xBuf_.Get<T>();
    LocalTensor<int32_t> targetLocal = targetBuf_.Get<int32_t>();
    LocalTensor<int32_t> isTargetLocal = isTargetBuf_.Get<int32_t>();
    LocalTensor<float> xFloat = xFloatBuf_.Get<float>();

    const int64_t xBlock = 32 / static_cast<int64_t>(sizeof(T));
    T xPadVal;
    if constexpr (sizeof(T) == 4) {
        *reinterpret_cast<uint32_t*>(&xPadVal) = 0u;
    } else {
        *reinterpret_cast<uint16_t*>(&xPadVal) = 0u;
    }

    // target 整行进 UB（截断前缀不可拆核）
    const int64_t tCopy = targetTile_ < targetLen_ ? targetTile_ : targetLen_;
    const int64_t tPad = AlignUp(tCopy, 8) - tCopy;
    DataCopyExtParams tParams = {1, static_cast<uint32_t>(tCopy * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> tPadParams = {true, 0, static_cast<uint8_t>(tPad), 0};
    DataCopyPad(targetLocal, targetGm_[row * targetLen_], tParams, tPadParams);
    event_t evtT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evtT);
    WaitFlag<HardEvent::MTE2_S>(evtT);

    // ---- 趟 1：按 classTile 写 is_target（计数型）----
    for (int64_t c0 = 0; c0 < classNum_; c0 += classTile_) {
        const int64_t tile = (c0 + classTile_ <= classNum_) ? classTile_ : (classNum_ - c0);
        for (int64_t c = 0; c < tile; ++c) {
            isTargetLocal.SetValue(c, 0);
        }
        for (int64_t k = 0; k < tCopy; ++k) {
            const int32_t t = targetLocal.GetValue(k);
            if (t < 0) {
                break;
            }
            if (t >= c0 && t < c0 + tile) {
                const int64_t local = static_cast<int64_t>(t) - c0;
                isTargetLocal.SetValue(local, isTargetLocal.GetValue(local) + 1);
            }
        }
        SyncSMTE3();
        DataCopyExtParams outParams = {1, static_cast<uint32_t>(tile * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(isTargetGm_[row * classNum_ + c0], isTargetLocal, outParams);
    }

    // ---- 趟 2：按 (j-tile × l-tile) 累加 hinge；中间 fp32 ----
    float loss = 0.0f;
    for (int64_t j0 = 0; j0 < classNum_; j0 += classTile_) {
        const int64_t jTile = (j0 + classTile_ <= classNum_) ? classTile_ : (classNum_ - j0);
        const int64_t jPad = AlignUp(jTile, xBlock) - jTile;
        DataCopyExtParams xParams = {1, static_cast<uint32_t>(jTile * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> xPadParams = {true, 0, static_cast<uint8_t>(jPad), xPadVal};
        DataCopyPad(xLocal, xGm_[row * classNum_ + j0], xParams, xPadParams);

        DataCopyExtParams cntParams = {1, static_cast<uint32_t>(jTile * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> cntPadParams = {
            true, 0, static_cast<uint8_t>(AlignUp(jTile, 8) - jTile), 0};
        DataCopyPad(isTargetLocal, isTargetGm_[row * classNum_ + j0], cntParams, cntPadParams);

        event_t evtJ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(evtJ);
        WaitFlag<HardEvent::MTE2_S>(evtJ);

        for (int64_t jj = 0; jj < jTile; ++jj) {
            xFloat.SetValue(jj, LoadXAsFloat(xLocal, jj));
        }

        for (int64_t jj = 0; jj < jTile; ++jj) {
            if (isTargetLocal.GetValue(jj) != 0) {
                continue; // 只对 cnt[j]==0 的非目标类
            }
            const float xj = xFloat.GetValue(jj);
            for (int64_t l0 = 0; l0 < classNum_; l0 += classTile_) {
                const int64_t lTile = (l0 + classTile_ <= classNum_) ? classTile_ : (classNum_ - l0);
                // 复用 xBuf_/isTargetBuf_ 加载正类块（j 块已在标量寄存器 xj）
                const int64_t lPad = AlignUp(lTile, xBlock) - lTile;
                DataCopyExtParams xlParams = {1, static_cast<uint32_t>(lTile * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> xlPadParams = {true, 0, static_cast<uint8_t>(lPad), xPadVal};
                DataCopyPad(xLocal, xGm_[row * classNum_ + l0], xlParams, xlPadParams);

                DataCopyExtParams clParams = {1, static_cast<uint32_t>(lTile * sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> clPadParams = {
                    true, 0, static_cast<uint8_t>(AlignUp(lTile, 8) - lTile), 0};
                // 注意：isTargetLocal 仍持有 j 块计数；正类 cnt 改从 GM 标量读，避免冲掉 j 掩码
                event_t evtL = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(evtL);
                WaitFlag<HardEvent::MTE2_S>(evtL);

                for (int64_t ll = 0; ll < lTile; ++ll) {
                    const int32_t cnt = isTargetGm_.GetValue(row * classNum_ + l0 + ll);
                    if (cnt == 0) {
                        continue;
                    }
                    const float xl = LoadXAsFloat(xLocal, ll);
                    const float pair = 1.0f + xj - xl;
                    // 逻辑比较：pair<=0 才丢弃；NaN 比较为假 → 保留（对齐 np.maximum）
                    if (!(pair <= 0.0f)) {
                        loss += static_cast<float>(cnt) * pair;
                    }
                }
                (void)clParams;
                (void)clPadParams;
            }
        }
    }
    loss = loss / static_cast<float>(classNum_);
    return loss;
}

template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::Process()
{
    if (rowCount_ <= 0 || classNum_ <= 0 || targetLen_ <= 0) {
        return;
    }

    for (int64_t i = 0; i < rowCount_; ++i) {
        const int64_t row = rowStart_ + i;
        const float loss = ProcessOneRow(row);
        rowLossGm_.SetValue(row, loss);
        if (reduction_ == REDUCTION_NONE) {
            StoreYValue(row, loss);
            SyncSMTE3();
        }
    }

    SyncAll();

    if (reduction_ != REDUCTION_NONE && GetBlockIdx() == 0) {
        float acc = 0.0f;
        for (int64_t row = 0; row < batchSize_; ++row) {
            acc += rowLossGm_.GetValue(row);
        }
        if (reduction_ == REDUCTION_MEAN && batchSize_ > 0) {
            acc = acc / static_cast<float>(batchSize_);
        }
        StoreYValue(0, acc);
        SyncSMTE3();
    }
}

} // namespace NsMultilabelMarginLoss
#endif // MULTILABELMARGINLOSS_H
