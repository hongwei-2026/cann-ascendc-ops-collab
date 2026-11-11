# DEC006 — 关闭P007连续行批量DMA

2026-11-11；hongwei-2026；ConfusionTransposeD_2323。

## 证据

- C007在910B3/CANN9.1.0 exit=0；P004与P007均完成`build.sh -e`，各39/39 bitwise PASS。
- 四项profile均成功识别25次Kernel调用，输出精确；下列数值为msprof kernel task中位数（5次warmup后20个样本）。

| 案例 | P004 (μs) | P007 (μs) | P004/P007 |
|---|---:|---:|---:|
| rank2_swap_f32 | 8.30 | 7.82 | 1.061× |
| baseline_tf_f32 | 4.08 | 4.36 | 0.936× |
| dma_rows128_f32 | 6.72 | 6.56 | 1.024× |
| larger_4d_f16 | 11.16 | 10.80 | 1.033× |

- 目标`dma_rows128_f32`要求至少1.3×，实际仅1.024×；小shape要求回退≤2%，实际耗时增加6.86%。
- 用户未能下载原始tar；只归档粘贴的marker段`explorations/logs/EXP_C007/C007-console-evidence.log`，SHA256 `816dfb66e398b839b5a9f2fc573ad3668a045b64abad774d70bccd1c97ea2911`。这是终端证据，不宣称完整原始日志已回收。

## 决定

P007不采纳并关闭。Kernel、Host tiling和tiling data已恢复为已验证P004提交`4761239`；P007脚本和记录仅作为实验历史保留。当前性能目标仍未完成，平台Runtime Error仍未定位。

依DEC005的止损条件，不继续修补或自动开启P008；需Owner明确授权新路线并先登记新的path-ledger行及时间预算后，才可继续性能实现。
