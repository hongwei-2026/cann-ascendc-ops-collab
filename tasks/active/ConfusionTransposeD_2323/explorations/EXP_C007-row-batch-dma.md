# EXP_C007 — P007 连续行批量读取候选

2026-11-11；ConfusionTransposeD_2323；hongwei-2026。P007已先登记path-ledger；基线P004=`4761239`。依DEC005，仅允许这一条90分钟有界跟进。

## 假设与实现范围

C006中P004的`dma_rows128_f32`基线为7.170μs；P004 `ProcessCopyRows`对每条输入连续行单独发起一次MTE2 `DataCopyPad`。当一组输出行对应的源行起点等差、源跨度有效，且行字节数为32B对齐时，单次多块`DataCopyPad`可替代多次逐行发令。目标≥1.3×，rank2和小shape回退不超过2%。

仅改本题Kernel路径并移除P006未采纳的Host/Kernel分派字段；保留P004 `copySpan`路径和其他布局回退。分组在源地址不连续、块跨度溢出或非32B行时重新走P004逐行处理，不拓展超UB、跨rank或dtype支持。

## 本地验证

`p007_semantic_check.py`以原reference逐字节比较TF/RF、perm/rank、tile/core边界，并核对每输出元素恰写一次；固定种子的随机原始字节避免周期性输入掩盖地址错误。266项本地计划通过，目标`dma_rows128_f32`模型中的MTE2调用从128降至32。该CPU oracle不验证CANN接口语义、NPU编译或性能。

## 云端验收

Owner按[`C007-CLOUD-RUN.md`](../C007-CLOUD-RUN.md)执行P004/候选隔离build、39项bitwise和4项profile。完整tar与终端证据均需归档；分析时只把msprof task time称为Kernel耗时，平台Runtime Error仍未确认。

截止：本地实现/语义≤90分钟；Owner回传后分析/必要修复≤30分钟。任一编译/精度失败经一次修复仍未解决，或性能门槛未达到，依据DEC005停止并恢复P004。
