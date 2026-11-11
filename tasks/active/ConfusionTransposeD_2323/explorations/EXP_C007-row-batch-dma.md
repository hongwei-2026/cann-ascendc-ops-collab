# EXP_C007 — P007 连续行批量读取候选

2026-11-11；ConfusionTransposeD_2323；hongwei-2026。P007已先登记path-ledger；基线P004=`4761239`。依DEC005，仅允许这一条90分钟有界跟进。

## 假设与实现范围

C006中P004的`dma_rows128_f32`基线为7.170μs；P004 `ProcessCopyRows`对每条输入连续行单独发起一次MTE2 `DataCopyPad`。当一组输出行对应的源行起点等差、源跨度有效，且行字节数为32B对齐时，单次多块`DataCopyPad`可替代多次逐行发令。目标≥1.3×，rank2和小shape回退不超过2%。

仅改本题Kernel路径并移除P006未采纳的Host/Kernel分派字段；保留P004 `copySpan`路径和其他布局回退。分组在源地址不连续、块跨度溢出或非32B行时重新走P004逐行处理，不拓展超UB、跨rank或dtype支持。

## 本地验证

`p007_semantic_check.py`以原reference逐字节比较TF/RF、perm/rank、tile/core边界，并核对每输出元素恰写一次；固定种子的随机原始字节避免周期性输入掩盖地址错误。266项本地计划通过，目标`dma_rows128_f32`模型中的MTE2调用从128降至32。该CPU oracle不验证CANN接口语义、NPU编译或性能。

## 云端结果与结案

2026-11-11，910B3/CANN9.1.0，C007脚本exit=0。P004与P007均`build.sh -e`完成，39/39 bitwise PASS；四项profile均为25次有效Kernel调用、输出字节精确一致。

| 案例 | P004 (μs) | P007 (μs) | P004/P007 |
|---|---:|---:|---:|
| rank2_swap_f32 | 8.30 | 7.82 | 1.061× |
| baseline_tf_f32 | 4.08 | 4.36 | 0.936× |
| dma_rows128_f32 | 6.72 | 6.56 | 1.024× |
| larger_4d_f16 | 11.16 | 10.80 | 1.033× |

主目标仅提升2.44%，远低于1.3×；小shape耗时增加6.86%，超过2%回退上限。按DEC006关闭P007，并将Kernel、Host tiling及tiling data精确恢复至P004提交`4761239`。本次只归档粘贴的C007 marker证据段，SHA256为`816dfb66e398b839b5a9f2fc573ad3668a045b64abad774d70bccd1c97ea2911`；原始tar未下载，故不称完整原始日志归档。平台Runtime Error仍未知，C007同机profile不代表平台用时。
