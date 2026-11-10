# 新对话提示词（进行中 · 选开）

日期：2026-11-10  
用途：开新对话时整段粘贴。  
不含：`PReluGradUpdate_2329`、`MultilabelMarginLoss_2327`、`tasks/archive/` 三题。

本机若落后远端，先在提示词里要求的 `git pull` 会拉到最新 STATUS。

---

## ConfusionTransposeD_2323

**完整性能交接提示词（推荐整段粘贴）：**  
[`tasks/active/ConfusionTransposeD_2323/HANDOFF-PERF-PROMPT.md`](../../tasks/active/ConfusionTransposeD_2323/HANDOFF-PERF-PROMPT.md)

```text
【新对话 · ConfusionTransposeD_2323 · 性能专题 · 勿串其它题】

题名：ConfusionTransposeD_2323
成员登录名：hongwei-2026
本会话唯一目标：有界性能优化（先 path-ledger 再改 Kernel）；保 EXP_C002 功能不回退。

已知（以 STATUS 为准，需复核）：
- 代码座位：people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/
- 状态：性能中；功能 C001/C002 PASS；平台 ~1.7ms vs 最优 ~3.5μs（~500×，EXP_C003）
- 瓶颈假设：ComputeTile 标量 GetValue/SetValue gather
- 交接全文：tasks/active/ConfusionTransposeD_2323/HANDOFF-PERF-PROMPT.md
- 止损：docs/collaboration/02-perf-gate.md

硬规则：只碰本题；只写 people/hongwei-2026/；启用 cann-spmv-ascendc skill；无 ledger 新行不换路。

开工：git pull → 读 STATUS/path-ledger/HANDOFF-PERF-PROMPT → 汇报假设与时限 → 等下令或直接按提示词 A/B 开工。
仓库：D:\sageAttention2 算子开发任务书
```

---

## CtcLossV2_2325

```text
【新对话 · CtcLossV2_2325 · 勿串其它题】

题名：CtcLossV2_2325
成员登录名：hongwei-2026
本会话唯一目标：pull 后摸清现状并汇报下一闸；未授权不改代码。

已知（需复核）：
- Owner/代码：people/yuhongming-2026/CtcLossV2_2325/workdir/code/
- 状态：实现中；本地 910B4 build.sh -e 样例 PASS（loss 0/1 · alpha 0/170，EXP_C001；TIK 语义）
- 台账：tasks/active/CtcLossV2_2325/（含 AI-SESSION / DIALOG）
- 下一闸：泛化 / 云测

硬规则：
1. 只碰本题；禁止改其它 active 题。
2. 先读 AGENTS.md、05-ai-work-protocol.md、CLAIM.md。
3. 不写 people/<他人>/；当前在 yuhongming 座位 → 先只读，授权后再开 hongwei 座位或指定改动范围。
4. 台账/探索/云日志按协议落盘并 push。
5. 启用 cann-spmv-ascendc skill。

开工：git pull → 读 STATUS/path-ledger/EXP → 汇报 → 等我下令。
仓库：D:\sageAttention2 算子开发任务书
```

---

## LayerNormV3_2326

```text
【新对话 · LayerNormV3_2326 · 勿串其它题】

题名：LayerNormV3_2326
成员登录名：hongwei-2026
本会话唯一目标：pull 后摸清现状并汇报下一闸；未授权不改代码。

已知（需复核）：
- Owner/代码：people/yuhongming-2026/LayerNormV3_2326/workdir/code/
- 状态：实现中；910B2 样例 + 一组泛化已过（y/mean/rstd 0 mismatch；含非对齐/多核/fp16/bf16/gamma=float）
- 本地语义：EXP_L001 ALL PASSED
- 台账：tasks/active/LayerNormV3_2326/
- 下一闸：平台 Judge 隐藏用例复测
- 近期 tip：a77cf02 等 LayerNorm 修复

硬规则：
1. 只碰本题；禁止串 PRelu / Multilabel 等。
2. 先读 AGENTS.md、05-ai-work-protocol.md、CLAIM.md。
3. 不写他人 people/；先只读汇报，授权后再动。
4. 台账/探索/云日志按协议落盘并 push。
5. 启用 cann-spmv-ascendc skill。

开工：git pull → 读 STATUS/path-ledger/EXP_L* → 汇报 → 等我下令。
仓库：D:\sageAttention2 算子开发任务书
```

---

## PadV3Grad_2328

```text
【新对话 · PadV3Grad_2328 · 勿串其它题】

题名：PadV3Grad_2328
成员登录名：hongwei-2026
本会话唯一目标：pull 后摸清现状并汇报下一闸；未授权不改代码。

已知（需复核）：
- Owner/代码：people/yuhongming-2026/PadV3Grad_2328/workdir/code/
- 状态：实现中；local_semantic PASS（EXP_P001）；NPU build.sh -e 未跑
- 台账：tasks/active/PadV3Grad_2328/（TASK.md / STATUS / SUBMIT_LINKS）
- 要点：按 reference 做 Pad 梯度折回；constant 按行分核，其它模式单核 float32；dtype fp16/fp32/bf16
- 下一闸：910B 上 bash build.sh -e
- tip 附近：2124a11

硬规则：
1. 只碰本题；禁止串其它题。
2. 先读 AGENTS.md、05-ai-work-protocol.md、CLAIM.md。
3. 不写他人 people/；先只读汇报，授权后再动。
4. 未跑通 -e 不得写成设备精度 PASS。
5. 台账/探索/云日志按协议落盘并 push；启用 cann-spmv-ascendc skill。

开工：git pull → 读 TASK/STATUS/path-ledger/EXP → 汇报 → 等我下令。
仓库：D:\sageAttention2 算子开发任务书
```
