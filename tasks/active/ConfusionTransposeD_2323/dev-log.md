# Dev log — ConfusionTransposeD_2323

## 2026-11-10（P004 候选，待云测）

- 同步 main=545f8d2；只在隔离副本的 hongwei 本题座位实现；补事实 SUMMARY/会话卡片，先登记 P004 连续行 DMA 假设与时限。
- 连续后缀走 MTE2→MTE3 整行批量 DMA，短行 UB 补齐后密集写出；span=1 保留原实现；超 UB tensor 边界不扩大。
- CPU local semantic + 1658 项 copy-plan PASS；22 项 fixture 生成、脚本语法检查通过。**无 CANN/NPU PASS 或性能收益结论**。
- 云测由 Owner 手动执行；一键脚本对比 545f8d2/候选，失败也打包；见 C004-CLOUD-RUN.md。下一步回收云日志、按时限结案/DEC，禁止无证据转路线。

## 2026-11-10（性能交接）

- 平台：点 1～5 功能 Pass、错误 0%；用时 ~1.7ms vs 最优 ~3.5μs（~500×）→ STATUS **性能中**
- 记 EXP_C003 基线；path-ledger **P003**；写 `HANDOFF-PERF-PROMPT.md` 供其他 AI 有界优化
- 根因假设：Kernel 标量 `GetValue`/`SetValue` gather；下一 AI 按提示词开 A/B 路，勿无台账死磕

## 2026-11-10

- 授权：拷贝座位到 `people/hongwei-2026/ConfusionTransposeD_2323/`，先做扩泛化/云测（不改 yuhongming workdir）
- path-ledger **P002** 开工；加泛化/云测脚本 + `SYNC_CODEARTS.md`；`207ea78` push
- **EXP_C002 云测 PASS**：exit 0；样例 0/120；回归 13/13；日志归档 `explorations/logs/EXP_C002*`
- 原「下一闸超 UB」后置，优先性能

## 2026-11-04

- 从 `ConfusionTransposeD_problem_2323_template.zip` 建座位与台账
- 实现 infershape / tiling / kernel（输出下标 → 输入下标）
- 本地 910B4 `bash build.sh -e`：**PASSED**（y mismatched=0/120）
