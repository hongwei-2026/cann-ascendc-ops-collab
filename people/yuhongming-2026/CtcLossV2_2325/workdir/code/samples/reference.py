#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""CTCLossV2 的 CPU 预期实现：内置 TIK 算子（ops_legacy/dynamic/ctc_loss_v2.py）的逐位模拟。

910B 上 CTCLossV2 走 ops_legacy 的 TIK 实现（kernel/config/ascend910b/ops_legacy/
ctc_loss_v2.json 预编译二进制）。该实现与标准 CTC 有若干实质差异，标准公式直接照抄
必然对不上，实测（探针 P1~P4 + 50 条用例的全部输出 bin）确认如下语义并逐条复刻：

1. 行内窗口：第 t 行只计算状态 [start(t), end(t))，窗口外保持 -inf。
   - end(0)=2，随 t 增长 e_inc[t-1]（t <= s_i + repeats 时）；
   - start 初值 0（repeats < t_i - s_i）否则 1，之后当 remain = s_i + repeats - t_i + t >= 0
     时按 s_inc[remain] 累加；
   - start==0 时状态 0 单独计算（prev[0]+lp[blank]），循环从 1 开始。
   递推本身允许 u-2 跳转（u 为奇数且当前标签与前一标签不同），u=0/1 有专门分支。
2. 缓冲布局错位：每样本 log_alpha 基址 = n * T * align8(2*s_max+1)（alpha_size_up），
   而行内步长是 output_size=2*s_max+1、每行写 output_size 个 float。s>=1 时
   align8 恒大于 2s+1（奇数），因此 n>=1 的样本整体后移，超出声明张量的部分**写飞**；
   未写入区域保留设备内存原值（msopst 输出缓冲复用上一条用例的数据，实测如此）。
   → 用例设计上必须 N=1 且 input_lengths=T，缓冲被完全覆盖，输出才确定。
3. t_i < s_i 的样本不写 log_alpha，loss = inf（zero_infinity=False）或 0（True）。
4. loss = -logadd(最后两个状态)，含 all(-inf) -> +inf 的怪路径；logadd 中间量按
   fp32 exp/ln 计算（-inf - -inf = nan 会传播）。
5. targets 值 < 0 时按 blank 类取概率；状态 s 的标签索引 = s//2，越界处不影响
   输出范围（>=output_size 的状态不会被搬出）。
"""
import numpy as np

NEG_INF = -np.inf
INF = np.inf
FP32_MAX = 3.4e38


def _count_trace(s_i, tgt, size):
    """复刻 count_trace：s_inc/e_inc 决定窗口推进；相邻同标签记 repeats（禁止跳转）。

    kernel 里两个数组按 output_size = 2*s_max+1 分配（不是 s_i+2）：
    repeats 多时 idx_counter 可达 2*s_i-1，窗口读到的每个槽都 <=
    idx_counter-1，即全是被写过的槽，所以只需按该上界分配即可，不会读到未初始化值。
    """
    s_inc = np.zeros(size, dtype=np.int64)
    e_inc = np.zeros(size, dtype=np.int64)
    repeats = 0
    idx_counter = 1
    s_inc[0] = 1
    if s_i >= 1:
        for idx in range(1, s_i):
            left = int(tgt[idx - 1])
            right = int(tgt[idx])
            if left == right:
                s_inc[idx_counter] = 1
                e_inc[idx_counter - 1] = 1
                s_inc[idx_counter + 1] = 1
                e_inc[idx_counter] = 1
                idx_counter += 2
                repeats += 1
            else:
                s_inc[idx_counter] = 2
                e_inc[idx_counter - 1] = 2
                idx_counter += 1
    e_inc[idx_counter - 1] = 1
    return repeats, s_inc, e_inc


def _logadd3(l1, l2, l3):
    """复刻 TIK 的三路 logadd：h_max -> h_sub -> h_exp -> h_add -> h_ln -> h_add。
    all(-inf) 时 -inf-(-inf)=nan，nan 一路传播（与实测 bin 中的 nan 一致）。"""
    lamax = max(l1, l2, l3)
    with np.errstate(all='ignore'):
        return np.log(np.exp(l1 - lamax) + np.exp(l2 - lamax) + np.exp(l3 - lamax)) + lamax


def _loss_from(a, b, zero_infinity):
    """复刻算子尾部 loss 计算（含全部 -inf / nan / +inf 的特殊分支）。"""
    if a > b:
        mx, u0, u1 = a, 0.0, b - a
    elif a == NEG_INF and b == NEG_INF:
        mx, u0, u1 = 0.0, a, b
    else:
        mx, u0, u1 = b, a - b, 0.0
    with np.errstate(all='ignore'):
        val = -np.log(np.exp(u0) + np.exp(u1)) - mx
    is_nan = (not (val <= FP32_MAX)) and (not (val > FP32_MAX))
    if (val < INF) or (not zero_infinity) or is_nan:
        return val
    return 0.0


def impl(log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False,
         reduction="mean"):
    """reduction 只作接口对齐，不参与计算：内置 TBE impl（ops_legacy/ctc_loss_v2.py）
    同样收下不使用（其 CTCLossV2 类不读该参数），910B 预编译二进制就是按
    reduction="mean" 编译的（kernel config simplifiedKey 里写死 ",mean,0"）；
    用例 N 恒为 1，mean/sum/none 在 N=1 时结果也相同。"""
    log_probs_np = np.asarray(log_probs, dtype=np.float64)
    targets_np = np.asarray(targets, dtype=np.int64)
    if targets_np.ndim == 1:
        targets_np = targets_np.reshape(1, -1) if log_probs_np.shape[1] == 1 else targets_np.reshape(-1, 1)
    input_lengths_np = np.asarray(input_lengths).astype(np.int64).reshape(-1)
    target_lengths_np = np.asarray(target_lengths).astype(np.int64).reshape(-1)

    T, N, C = log_probs_np.shape
    s_max = int(np.max(target_lengths_np))
    # 内置 log_alpha 末维 = 2*max(target_lengths)+1（不是 2*S+1），单独用 s_max 计算
    U = 2 * s_max + 1
    up = (U + 7) // 8 * 8
    alpha_size_up = T * up

    batch_loss = np.zeros(N, dtype=np.float64)
    buf = np.zeros(N * T * U, dtype=np.float64)

    for n in range(N):
        t_i = int(input_lengths_np[n])
        s_i = int(target_lengths_np[n])
        if t_i < s_i:
            batch_loss[n] = 0.0 if zero_infinity else INF
            continue

        tgt = targets_np[n]
        repeats, s_inc, e_inc = _count_trace(s_i, tgt, U)
        lpr = log_probs_np[:, n, :]

        # 第 0 行：状态 0/1 直接取 log_probs，其余 -inf
        dst = np.full(up, NEG_INF, dtype=np.float64)
        src = np.full(up, NEG_INF, dtype=np.float64)
        dst[0] = lpr[0, blank]
        ct = int(tgt[0])
        dst[1] = lpr[0, ct] if ct >= 0 else lpr[0, blank]

        start = 0 if repeats < t_i - s_i else 1
        end = 2
        written = {}
        for t in range(1, t_i):
            src[:] = NEG_INF
            remain = s_i + repeats - t_i + t
            if remain >= 0:
                start += int(s_inc[remain])
            start_loop = start
            if t <= s_i + repeats:
                end += int(e_inc[t - 1])
            if start_loop == 0:
                src[0] = dst[0] + lpr[t, blank]
                start_loop = 1
            for s in range(start_loop, end):
                if s % 2 == 0:
                    ct = blank
                else:
                    ct = int(tgt[s // 2])
                clp = lpr[t, ct] if ct >= 0 else lpr[t, 0]
                l1 = dst[s]
                l2 = dst[s - 1]
                if s % 2 != 0 and s != 1:
                    nt = int(tgt[s // 2 - 1])
                    l3 = dst[s - 2] if ct != nt else NEG_INF
                else:
                    l3 = NEG_INF
                src[s] = _logadd3(l1, l2, l3) + clp
            # 第 t-1 行在 dst 槽里，按 output_size 长写出
            written[t - 1] = dst[:U].copy()
            dst, src = src, dst
        written[t_i - 1] = dst[:U].copy()

        # 每样本基址 n*T*align8(U)（内置布局错位点；越界部分写飞，不到输出里）
        base = n * alpha_size_up
        for r, vals in written.items():
            off = base + r * U
            if off >= buf.size:
                break
            take = min(U, buf.size - off)
            buf[off:off + take] = vals[:take]

        a = dst[2 * s_i]
        b = dst[2 * s_i - 1] if s_i > 0 else NEG_INF
        batch_loss[n] = _loss_from(a, b, zero_infinity)

    loss_res = batch_loss.astype(np.float32)
    log_alpha_res = buf.reshape(N, T, U).astype(np.float32)
    return loss_res, log_alpha_res
