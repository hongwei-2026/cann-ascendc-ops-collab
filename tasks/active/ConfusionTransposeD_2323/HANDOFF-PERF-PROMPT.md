# 交接提示词 — ConfusionTransposeD 性能专题（给其他 AI）

> 整段复制到新对话即可。仓库规范优先于口头发挥。

```text
【新对话 · ConfusionTransposeD_2323 · 性能专题 · 勿串其它题】

题名：ConfusionTransposeD_2323
成员登录名：hongwei-2026
本会话唯一目标：有界性能优化（先建 path-ledger 行 + 假设时限，再改 Kernel）；保功能/精度不回退。

======== 背景（已发生事实）========
1. 功能已通：
   - EXP_C001：本地 910B4 样例 build.sh -e → y mismatched=0/120
   - EXP_C002：云测样例 PASSED + npu_regression 13/13 PASS（TF 真/假、rank2/3/4、同 rank reshape、f32/f16/bf16/i8/i32）
2. 平台提交：测试点 1～5 功能 Pass、输出错误占比 0.00%；点 6 隐藏。
3. 性能极差（平台对照表，约 2026-11-10）：
   | 点 | 我方用时 | 最优用时 |
   |----|----------|----------|
   | 1  | 1.76ms   | 3.66μs   |
   | 2  | 1.75ms   | 3.45μs   |
   | 3  | 1.73ms   | 3.45μs   |
   | 4  | 1.70ms   | 3.50μs   |
   | 5  | 1.82ms   | 3.43μs   |
   约慢 **~480–530×**（ms vs μs）。功能绿但门禁分数会被性能拖死。

======== 影响 ========
- 当前实现可合入功能，但无法接近 TBE/平台最优；继续用标量 gather 死磕无意义。
- 最优用时在 μs 量级 → 平台 case 很可能是小/中 shape（perf-gate 可能落在 B 类小 shape，但仍需先证明能否到同量级，再决定豁免材料 vs 继续优化）。
- 超 UB 大 tensor 分块仍是功能债，但**本闸优先打性能**；大改勿与超 UB 分块无台账缠在一起。

======== 现状实现（瓶颈根因）========
代码座位（只改这里）：
  people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/

核心 Kernel：op_kernel/confusion_transpose_d.h
方案（P001 采纳）：
  - tiling：整输入常驻 UB；超 UB 直接 tiling fail
  - Process：CopyInFull → 按 ubFactor 切输出 tile
  - ComputeTile：**for 循环 + MapSrcIndex + GetValue/SetValue 标量搬运**
    → 这是主瓶颈（无向量化 gather、无连续块拷、无专用 Transpose API）
  - 多核：按 totalNum 切 blockFactor（小 shape 可能实际 1 核）

Host tiling：op_host/confusion_transpose_d_tiling.cpp
约束：len(perm)==len(shape)；dtype 按字节宽 → tilingKey 0/1/2/3

台账：tasks/active/ConfusionTransposeD_2323/
  STATUS / path-ledger / EXP_C001 / EXP_C002 / SYNC_CODEARTS.md / docs/collaboration/02-perf-gate.md

======== 问题定义（本会话要解决）========
在**不毁掉**样例与 EXP_C002 泛化精度的前提下，把平台可见 case 从 ~1.7ms 压向 μs 量级（至少先砍 10×～100×，再对标最优）。
每一条优化路必须：path-ledger 新行 → 有时限 → 云测/平台证据 → 结案；触止损写 DEC。

======== 硬规则 ========
1. 只碰 ConfusionTransposeD_2323；禁止改 PRelu / Multilabel / 其它题。
2. 先读 AGENTS.md、docs/collaboration/05-ai-work-protocol.md、02-perf-gate.md、people/CLAIM.md、本题 STATUS/path-ledger。
3. 代码只写 people/hongwei-2026/ConfusionTransposeD_2323/；禁止写 people/yuhongming-2026/ 等他人目录。
4. 启用 ~/.cursor/skills/cann-spmv-ascendc/SKILL.md。
5. 无 path-ledger 新行不得换优化路线；失败也要记 EXP + 云日志归档 push。
6. 禁止假 PASS；改完必须至少：local_semantic_check + 云端 build.sh -e +（尽量）npu_regression。
7. 小 shape 死磕触发 02-perf-gate 止损 → 写 decisions/DECxxx.md，勿无限抠。

======== 建议思路（按优先级试；每条单独开 ledger）========
A. 【高优先】消灭标量 GetValue/SetValue
   - 识别「末维连续 / 整行/整平面可块拷」的 perm，走 DataCopy（含 stride/跳拷）代替逐元素。
   - 常见 ConfusionTranspose 模式（如 [n,c,h,w]↔重排）特化 kernel + tiling 分派。
B. 【高优先】用 AscendC 向量/搬运原语做 gather/scatter 或 Transpose
   - 查本 CANN 版本是否有 Transpose / Gather / Copy with ndma 等；910B（任务书）路径，勿照搬 950 SIMT 除非确认环境。
C. 【中】双缓冲 + 多核真正吃满
   - BUFFER_NUM、CopyIn/Compute/CopyOut 流水；确认平台 case 的 BlockDim 与尾核。
D. 【中】小 shape 特化
   - 整 tensor 极小时：一次搬入、一次写出、减少 queue/同步开销；避免每 tile 反复 DeQue/EnQue 输入。
E. 【低/后置】超 UB 分块
   - 与性能正交；功能需要时另开 P 行，勿与 A/B 混在一个 diff。
F. 【对照】
   - 找 TBE ConfusionTransposeD / npu_confusion_transpose 同机基线；记录是否「小 shape ≤10μs 差 3μs 可走仿真豁免」。

废弃/勿重复：
- 不要在「仍用标量 MapSrcIndex 循环」上微调循环变量指望到 μs。
- 不要无证据改 dtype key / 空改 tiling 字段。

======== 验收口令 ========
- 功能：样例 -e PASS；回归尽量仍 13/13（或写明跳过的 dtype）。
- 性能：平台重提或同机 profile；相对本基线（~1.7ms）有数量级提升；对照最优 μs 写清差距与是否触止损。
- 台账：更新 STATUS（性能中）/ path-ledger / EXP_C00x / 必要时 DEC；日志进 explorations/logs/ 并 push。

开工：git pull origin main → 读 STATUS/path-ledger/HANDOFF-PERF-PROMPT → 写 path-ledger 新行假设与时限 → 最小 diff 试 A 或 B → 云测归档。
仓库：D:\sageAttention2 算子开发任务书
```
