# ConfusionTransposeD_2323 — 当前事实与性能围栏

2026-11-10 补录：此前无 SUMMARY；本页依据已归档 EXP 和 Owner 本次性能授权整理，不冒充多路分析或新增验收。

- 目标/座位：hongwei-2026；只在 `people/hongwei-2026/ConfusionTransposeD_2323/` 写实现，本题 `tasks/active/ConfusionTransposeD_2323/` 写台账。
- P001/P002 已采纳功能精度：EXP_C001 样例 0/120；EXP_C002 样例 PASS、NPU 泛化 13/13。此前日志不证明本次候选通过。
- 语义：TF=true 先按输入 shape 转置再 reshape；TF=false 先按 attr shape reshape 再转置。连续性必须由实际转置前 shape 判定，不能总用最终 shape。
- 当前边界：整输入 UB 方案，超 UB tiling fail；perm/shape 同 rank；本次不扩大超 UB 支持范围。
- 平台基线：EXP_C003 点 1～5 1.70～1.82 ms，最优 3.43～3.66 us；case shape、BlockDim 和同机 TBE 时间未知。不能据此宣称 B 类或套用其他题的 95%/3 us 豁免。
- 本次批准推进 P004（路线 A），先登记假设/时限，候选通过 local semantic + 云端 build -e + regression 后才考虑采纳；同机 msprof 对照才判断数量级收益。

2026-11-10 云日志回收：P004 已在 910B3/CANN9.1.0 通过样例及22/22 bitwise；三项连续行同机收益2.09～6.07×，rank2回退无收益。采纳有限收益、≥10×未达；按 DEC001 冻结Kernel，先平台重提。相同原版在本轮为8～41us，不能把平台1.7ms的差距全部归因标量循环，也不能用跨case跨环境时间宣称400×。

## 废弃/禁止重复

- 保留逐元素 MapSrcIndex + GetValue/SetValue，只微调循环变量。
- 无证据变更 dtype tilingKey、空改字段、引入 950 SIMT。
- 在本 diff 混入超 UB 分块、其他题目改动、他人座位写入。
- 无新 ledger 换路线；超时或无 profile 继续盲改；把平台最优视为已测同机 TBE。
