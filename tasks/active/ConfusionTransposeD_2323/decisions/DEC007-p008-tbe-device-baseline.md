# DEC007 — P008：系统 TBE device 基线已立；停 Host；禁微改 Kernel

- 日期：2026-11-12
- 题：ConfusionTransposeD_2323
- 座位：hongwei-2026
- 相关：P008 / EXP_C008 / 02-perf-gate

## 已证实

1. 系统可调用：`torch_npu.npu_confusion_transpose`，`ASCEND_CUSTOM_OPP_PATH` unset。
2. Host 多尺度 ~0.09ms 贴死 → **Host 不能当 kernel 基线**。
3. msprof device：核名 `te_confusiontransposed_*__kernel0`（TE/TBE）；`(2,1024,128)` f32 稳态 Task Duration **≈4.7–4.9μs**（末 10 次均值 ≈4.80μs）。
4. 与平台最优 ~3.5μs **同量级**；与 P004 平台点 ~717μs **差两个数量级**。
5. 座位 `examples/run.sh` 只测自定义 `ConfusionTransposeD`，不能当系统基线。

证据：`explorations/logs/EXP_C008/C008-msprof-system-paste.log` 及同目录 Host/符号粘贴。

## 决策

| 选项 | 结论 |
|------|------|
| 继续 P004 类 DMA/行合并微优化 | **否**（P006/P007 已止损；相对 TBE 结构性差距） |
| 用 Host 时间宣称逼近 TBE | **否** |
| 把平台最优当成「未知黑盒」 | **否**——证据指向系统 TE ConfusionTransposeD |
| 新 Kernel 路线 | **仅当** Owner 书面批准「镜像 TBE/arch35 分派模板」结构大改（≤2 人日，先读模板再编码）；否则 **性能专题暂停**，活动包保持 P004 |
| 平台粘贴未验证新包 | **禁止** |

## 下一动作

1. 补一行 `op_summary` 归档（可选，不阻塞）。
2. Owner 二选一：批准结构大改立项 / 性能阶段收口转平台 RE 与交付。
3. 未立项前 **不改** `people/hongwei-2026/.../op_kernel`。
