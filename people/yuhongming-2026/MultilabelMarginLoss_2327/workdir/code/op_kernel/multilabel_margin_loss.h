/*!
 * \file multilabel_margin_loss.h
 * \brief MultilabelMarginLoss AscendC kernel（对齐 910B TBE 计数语义）
 *
 * is_target：扫描到首个 <0 为止，对合法类标计数（int32）。
 * loss：对非目标类 j 与目标类 l，累加 cnt[l]*max(0, 1+x[j]-x[l]) 再 /C。
 * 中间一律 float32，最后按输出 dtype 舍入。
 *
 * ────────────────────────────────────────────────────────────────────────
 * 【EXP_M001 失败根因 · 已按 CANN 8.5.0 接口参考改正】
 *
 * 现象：is_target PASS(0/24)，y Wrong Answer(1/1)。
 *
 * 根因：旧版用 `rowLossGm_.SetValue()` / `yGm_.SetValue()` 写 GM、用
 *       `GetValue()` 跨核读回。手册 §4.6.2 明确：
 *         · DMA 搬运单元（DataCopy/DataCopyPad）读写 GM —— **没有 Cache 一致性问题**；
 *         · Scalar 单元访问 GM —— 先过每个核内部的 Data Cache，**存在一致性问题**。
 *       §3.2.7 SetValue 进一步规定：
 *         · 写只改 DCache，**不会立刻写出 GM**，写出以 Cache Line(64B) 为单位；
 *         · **多核操作 GM 时不同核地址至少要差一个 Cache Line，否则多核数据随机覆盖**。
 *       本算子 N=4 走 4 个核，各核写 rowLoss[0..3]，偏移仅 0/4/8/12B，
 *       **同处一条 64B Cache Line** → 各核回写整行时相互覆盖 → y 读到随机值。
 *       而 is_target 走 DataCopyPad（DMA）不过 DCache，所以一直是对的。
 *       这解释了「计数对、损失错」的全部现象。
 *
 * 改正：y / rowLoss 路径**全部改走 DataCopyPad（MTE3/MTE2 DMA）**，
 *       kernel 内不再出现任何对 GM 的标量 SetValue/GetValue。
 *
 * 另修（同次定位）：
 *   1. 旧版 Pass2 把 is_target 写出 GM 后又用 `isTargetGm_.GetValue()` 读回当 cnt，
 *      既是 MTE3→Scalar 的读后写冒险，又违反 §3.2.5 GetValue 约束（GM 内容可能
 *      被外部改写时须先 DataCacheCleanAndInvalid）。读回若拿到旧值则全行判为
 *      「无目标类」→ loss 恒 0。现改为**从常驻 UB 的 target 行就地重算 cnt**，
 *      彻底去掉 GM 往返。
 *   2. 旧版 `rowCount_<=0` 时提前 return，**跳过了 SyncAll()**。当
 *      ceil(N/rowsPerCore) < usedCores（如 N=100、coreNum=64 → 仅 50 核有活）
 *      时，空闲核不参与 SyncAll，其余核永久等待 → 死锁。现改为无条件参与。
 *   3. hinge 内层由标量改为**沿 j 轴向量化**，并按互异目标类 cnt 加权
 *      （等价于按前缀位置展开），由 O(C*D) 标量降为 O(D) 次向量操作。
 * ────────────────────────────────────────────────────────────────────────
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

constexpr int64_t REDUCTION_NONE = 0;
constexpr int64_t REDUCTION_MEAN = 1;
constexpr int64_t REDUCTION_SUM = 2;

// 跨核归约时每次从 workspace 读回的 float 数（UB 常驻 RED_CHUNK*4B）
constexpr int64_t RED_CHUNK = 512;
// y 单次写出的暂存元素数（只用首元素，留 32 个槽保证 32B 对齐余量）
constexpr int64_t OUT_SLOT = 32;

template <typename T>
class MultilabelMarginLoss {
public:
    __aicore__ inline MultilabelMarginLoss() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR y, GM_ADDR isTarget,
        GM_ADDR workspace, const MultilabelMarginLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t alignment) const
    {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    __aicore__ inline void LoadTargetRow(int64_t row);
    // 不可 const：TBuf::Get 非 const（CANN 9.1 AscendC）
    __aicore__ inline int64_t ScanPrefix();
    __aicore__ inline void BuildCnt(int64_t c0, int64_t len);
    __aicore__ inline void BuildMask(int64_t c0, int64_t len);
    __aicore__ inline void LoadXTile(int64_t row, int64_t c0, int64_t len,
        const LocalTensor<float>& dst, float bias);
    __aicore__ inline float ProcessOneRow(int64_t row);

    // 以下两个写出接口一律走 DataCopyPad（DMA），禁止标量写 GM
    __aicore__ inline void StoreRowLoss(int64_t row, float value);
    __aicore__ inline void StoreY(int64_t idx, float value);

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> tgtBuf_;   // int32  K
    TBuf<TPosition::VECCALC> xRawBuf_;  // T      classTile
    TBuf<TPosition::VECCALC> xjfBuf_;   // float  classTile  (= x_j + 1)
    TBuf<TPosition::VECCALC> xlfBuf_;   // float  classTile  (= x_l)
    TBuf<TPosition::VECCALC> maskBuf_;  // float  classTile  (非目标类 1.0 / 目标类 0.0)
    TBuf<TPosition::VECCALC> accBuf_;   // float  classTile  hinge 累加器
    TBuf<TPosition::VECCALC> tmpBuf_;   // float  classTile  scratch / ReduceSum work
    TBuf<TPosition::VECCALC> cntBuf_;   // int32  classTile  计数
    TBuf<TPosition::VECCALC> redBuf_;   // float  RED_CHUNK  归约读回 + ReduceSum dst
    TBuf<TPosition::VECCALC> outBuf_;   // T      OUT_SLOT   y 写出暂存

    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> targetGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> isTargetGm_;
    GlobalTensor<float> rowLossGm_;

    int64_t batchSize_ = 0;
    int64_t classNum_ = 0;
    int64_t targetLen_ = 0;
    int64_t reduction_ = REDUCTION_MEAN;
    int64_t rowsPerCore_ = 1;
    int64_t classTile_ = 0;
    int64_t rowStart_ = 0;
    int64_t rowCount_ = 0;
    int64_t prefixLen_ = 0;
};

template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::Init(GM_ADDR x, GM_ADDR target, GM_ADDR y,
    GM_ADDR isTarget, GM_ADDR workspace, const MultilabelMarginLossTilingData* tilingData)
{
    batchSize_ = tilingData->batchSize;
    classNum_ = tilingData->classNum;
    targetLen_ = tilingData->targetLen;
    reduction_ = tilingData->reduction;
    rowsPerCore_ = tilingData->rowsPerCore;
    classTile_ = tilingData->classTile > 0 ? tilingData->classTile : classNum_;
    if (classTile_ > classNum_) {
        classTile_ = classNum_;
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
    yGm_.SetGlobalBuffer((__gm__ T*)y, reduction_ == REDUCTION_NONE ? batchSize_ : 1);
    isTargetGm_.SetGlobalBuffer((__gm__ int32_t*)isTarget, batchSize_ * classNum_);
    rowLossGm_.SetGlobalBuffer((__gm__ float*)GetUserWorkspace(workspace), batchSize_);

    const int64_t xElemPerBlock = 32 / static_cast<int64_t>(sizeof(T));
    const int64_t cAlign = AlignUp(classTile_, 8);

    pipe_.InitBuffer(tgtBuf_, AlignUp(targetLen_, 8) * sizeof(int32_t));
    pipe_.InitBuffer(xRawBuf_, AlignUp(classTile_, xElemPerBlock) * sizeof(T));
    pipe_.InitBuffer(xjfBuf_, cAlign * sizeof(float));
    pipe_.InitBuffer(xlfBuf_, cAlign * sizeof(float));
    pipe_.InitBuffer(maskBuf_, cAlign * sizeof(float));
    pipe_.InitBuffer(accBuf_, cAlign * sizeof(float));
    pipe_.InitBuffer(tmpBuf_, cAlign * sizeof(float));
    pipe_.InitBuffer(cntBuf_, cAlign * sizeof(int32_t));
    pipe_.InitBuffer(redBuf_, RED_CHUNK * sizeof(float));
    pipe_.InitBuffer(outBuf_, OUT_SLOT * sizeof(T));
}

// target 整行常驻 UB —— 「首个负值截断」是顺序依赖的，前缀不可跨核/跨块拆分
template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::LoadTargetRow(int64_t row)
{
    LocalTensor<int32_t> tgt = tgtBuf_.Get<int32_t>();
    const int64_t pad = AlignUp(targetLen_, 8) - targetLen_;
    DataCopyExtParams params = {1, static_cast<uint32_t>(targetLen_ * sizeof(int32_t)), 0, 0, 0};
    // 尾部补 0（非负）：补负数会被误判成截断点，把有效前缀砍短
    DataCopyPadExtParams<int32_t> padParams = {true, 0, static_cast<uint8_t>(pad), 0};
    DataCopyPad(tgt, targetGm_[row * targetLen_], params, padParams);
    PipeBarrier<PIPE_ALL>(); // MTE2 -> Scalar
}

template <typename T>
__aicore__ inline int64_t MultilabelMarginLoss<T>::ScanPrefix()
{
    LocalTensor<int32_t> tgt = tgtBuf_.Get<int32_t>();
    for (int64_t k = 0; k < targetLen_; ++k) {
        if (tgt.GetValue(k) < 0) {
            return k;
        }
    }
    return targetLen_;
}

// 前缀内落在 [c0, c0+len) 的合法类标计数；>=C 的正值跳过但**不截断**
template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::BuildCnt(int64_t c0, int64_t len)
{
    LocalTensor<int32_t> tgt = tgtBuf_.Get<int32_t>();
    LocalTensor<int32_t> cnt = cntBuf_.Get<int32_t>();
    for (int64_t c = 0; c < len; ++c) {
        cnt.SetValue(c, 0);
    }
    for (int64_t k = 0; k < prefixLen_; ++k) {
        const int32_t t = tgt.GetValue(k);
        if (t < c0 || t >= c0 + len || t >= classNum_) {
            continue;
        }
        const int64_t idx = static_cast<int64_t>(t) - c0;
        cnt.SetValue(idx, cnt.GetValue(idx) + 1);
    }
}

// 非目标类掩码：cnt==0 → 1.0，cnt>0 → 0.0。只扫前缀，O(L) 标量，不需遍历 C
template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::BuildMask(int64_t c0, int64_t len)
{
    LocalTensor<int32_t> tgt = tgtBuf_.Get<int32_t>();
    LocalTensor<float> mask = maskBuf_.Get<float>();
    Duplicate(mask, 1.0f, static_cast<int32_t>(len));
    PipeBarrier<PIPE_ALL>(); // Vector -> Scalar
    for (int64_t k = 0; k < prefixLen_; ++k) {
        const int32_t t = tgt.GetValue(k);
        if (t < c0 || t >= c0 + len || t >= classNum_) {
            continue;
        }
        mask.SetValue(static_cast<int64_t>(t) - c0, 0.0f);
    }
    PipeBarrier<PIPE_ALL>(); // Scalar -> Vector
}

// 载入 x 的一个类别块并升 fp32；bias 用于一次性叠加 hinge 的 +1
template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::LoadXTile(int64_t row, int64_t c0, int64_t len,
    const LocalTensor<float>& dst, float bias)
{
    const int64_t xElemPerBlock = 32 / static_cast<int64_t>(sizeof(T));
    const int64_t pad = AlignUp(len, xElemPerBlock) - len;
    T padVal;
    if constexpr (sizeof(T) == 4) {
        *reinterpret_cast<uint32_t*>(&padVal) = 0u;
    } else {
        *reinterpret_cast<uint16_t*>(&padVal) = 0u;
    }

    if constexpr (std::is_same<T, float>::value) {
        DataCopyExtParams params = {1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams = {true, 0, static_cast<uint8_t>(pad), 0.0f};
        DataCopyPad(dst, xGm_[row * classNum_ + c0], params, padParams);
        PipeBarrier<PIPE_ALL>(); // MTE2 -> Vector
    } else {
        LocalTensor<T> raw = xRawBuf_.Get<T>();
        DataCopyExtParams params = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {true, 0, static_cast<uint8_t>(pad), padVal};
        DataCopyPad(raw, xGm_[row * classNum_ + c0], params, padParams);
        PipeBarrier<PIPE_ALL>(); // MTE2 -> Vector
        // half / bfloat16_t -> float 均为精确加宽
        Cast(dst, raw, RoundMode::CAST_NONE, static_cast<int32_t>(len));
        PipeBarrier<PIPE_V>();
    }
    if (bias != 0.0f) {
        Adds(dst, dst, bias, static_cast<int32_t>(len));
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline float MultilabelMarginLoss<T>::ProcessOneRow(int64_t row)
{
    LocalTensor<int32_t> cnt = cntBuf_.Get<int32_t>();
    LocalTensor<float> xjf = xjfBuf_.Get<float>();
    LocalTensor<float> xlf = xlfBuf_.Get<float>();
    LocalTensor<float> mask = maskBuf_.Get<float>();
    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<float> red = redBuf_.Get<float>();

    LoadTargetRow(row);
    prefixLen_ = ScanPrefix();

    // ---- 趟 1：按 classTile 写 is_target（计数型，DataCopyPad 写出）----
    for (int64_t c0 = 0; c0 < classNum_; c0 += classTile_) {
        const int64_t len = (c0 + classTile_ <= classNum_) ? classTile_ : (classNum_ - c0);
        BuildCnt(c0, len);
        PipeBarrier<PIPE_ALL>(); // Scalar -> MTE3
        DataCopyExtParams params = {1, static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(isTargetGm_[row * classNum_ + c0], cnt, params);
        PipeBarrier<PIPE_ALL>(); // MTE3 完成后方可复用 cntBuf_
    }

    // ---- 趟 2：沿 j 轴向量化 hinge，按互异目标类 cnt 加权 ----
    float loss = 0.0f;
    for (int64_t j0 = 0; j0 < classNum_; j0 += classTile_) {
        const int64_t jLen = (j0 + classTile_ <= classNum_) ? classTile_ : (classNum_ - j0);

        LoadXTile(row, j0, jLen, xjf, 1.0f); // xjf = x_j + 1
        BuildMask(j0, jLen);                 // mask = 1.0 iff cnt[j]==0
        Duplicate(acc, 0.0f, static_cast<int32_t>(jLen));
        PipeBarrier<PIPE_V>();

        for (int64_t l0 = 0; l0 < classNum_; l0 += classTile_) {
            const int64_t lLen = (l0 + classTile_ <= classNum_) ? classTile_ : (classNum_ - l0);
            LoadXTile(row, l0, lLen, xlf, 0.0f); // xlf = x_l
            BuildCnt(l0, lLen);
            PipeBarrier<PIPE_ALL>(); // Scalar(cnt) / Vector(xlf) -> Scalar 读取

            for (int64_t ll = 0; ll < lLen; ++ll) {
                const int32_t c = cnt.GetValue(ll);
                if (c == 0) {
                    continue; // 只对目标类 l
                }
                const float xl = xlf.GetValue(ll);
                PipeBarrier<PIPE_ALL>(); // Scalar -> Vector（xl 作为立即数）
                // tmp = max(0, 1 + x[j] - x[l])
                Adds(tmp, xjf, -xl, static_cast<int32_t>(jLen));
                PipeBarrier<PIPE_V>();
                Maxs(tmp, tmp, 0.0f, static_cast<int32_t>(jLen));
                PipeBarrier<PIPE_V>();
                if (c != 1) {
                    // 重复标签按出现次数加权（等价于按前缀位置逐个展开）
                    Muls(tmp, tmp, static_cast<float>(c), static_cast<int32_t>(jLen));
                    PipeBarrier<PIPE_V>();
                }
                Add(acc, acc, tmp, static_cast<int32_t>(jLen));
                PipeBarrier<PIPE_V>();
            }
        }

        // 目标类 j 自身的 max(0, 1+x[l]-x[l])=1 必须屏蔽，否则凭空多加
        Mul(acc, acc, mask, static_cast<int32_t>(jLen));
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(red, acc, tmp, static_cast<int32_t>(jLen));
        PipeBarrier<PIPE_ALL>(); // Vector -> Scalar
        loss += red.GetValue(0);
    }

    return loss / static_cast<float>(classNum_);
}

template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::StoreRowLoss(int64_t row, float value)
{
    LocalTensor<float> red = redBuf_.Get<float>();
    red.SetValue(0, value);
    PipeBarrier<PIPE_ALL>(); // Scalar -> MTE3
    DataCopyExtParams params = {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(rowLossGm_[row], red, params);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::StoreY(int64_t idx, float value)
{
    LocalTensor<T> out = outBuf_.Get<T>();
    if constexpr (std::is_same<T, float>::value) {
        out.SetValue(0, value);
        PipeBarrier<PIPE_ALL>(); // Scalar -> MTE3
    } else {
        LocalTensor<float> red = redBuf_.Get<float>();
        red.SetValue(0, value);
        PipeBarrier<PIPE_ALL>(); // Scalar -> Vector
        // fp32 -> fp16/bf16 取最近偶数，与 numpy astype 一致
        Cast(out, red, RoundMode::CAST_RINT, 1);
        PipeBarrier<PIPE_ALL>(); // Vector -> MTE3
    }
    DataCopyExtParams params = {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm_[idx], out, params);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void MultilabelMarginLoss<T>::Process()
{
    const bool hasWork = (rowCount_ > 0 && classNum_ > 0 && targetLen_ > 0);

    if (hasWork) {
        for (int64_t i = 0; i < rowCount_; ++i) {
            const int64_t row = rowStart_ + i;
            const float loss = ProcessOneRow(row);
            if (reduction_ == REDUCTION_NONE) {
                StoreY(row, loss);
            } else {
                StoreRowLoss(row, loss);
            }
        }
    }

    if (reduction_ == REDUCTION_NONE) {
        return; // none 各核独占行区间，无需跨核汇总
    }

    // 注意：无论本核是否有活都必须参与，否则空闲核缺席会让其余核永久等待
    SyncAll();

    if (GetBlockIdx() != 0) {
        return;
    }

    // 单核有序归约：按行序 fp32 累加，杜绝多核无序 AtomicAdd 的不可复现
    LocalTensor<float> red = redBuf_.Get<float>();
    float acc = 0.0f;
    for (int64_t base = 0; base < batchSize_; base += RED_CHUNK) {
        int64_t len = batchSize_ - base;
        if (len > RED_CHUNK) {
            len = RED_CHUNK;
        }
        const int64_t pad = AlignUp(len, 8) - len;
        DataCopyExtParams params = {1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams = {true, 0, static_cast<uint8_t>(pad), 0.0f};
        DataCopyPad(red, rowLossGm_[base], params, padParams);
        PipeBarrier<PIPE_ALL>(); // MTE2 -> Scalar
        for (int64_t i = 0; i < len; ++i) {
            acc += red.GetValue(i);
        }
    }

    if (reduction_ == REDUCTION_MEAN && batchSize_ > 0) {
        // 只在最后除一次 N；不与 /C 融合成 Σ/(N*C)，fp32 下两者不等价
        acc = acc / static_cast<float>(batchSize_);
    }
    StoreY(0, acc);
}

} // namespace NsMultilabelMarginLoss
#endif // MULTILABELMARGINLOSS_H
