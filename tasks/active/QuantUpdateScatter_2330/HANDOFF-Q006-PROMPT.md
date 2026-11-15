# QuantUpdateScatter_2330 — Q006 交接提示词

把下面「整段粘贴」框给其他 AI。仓库需先 `git pull` 到含本文件的 `main`（Q005 结案约 `361a83f`+）。

---

```text
【新对话 · QuantUpdateScatter_2330 · Q006 性能 · 勿串其它题】

题名：QuantUpdateScatter_2330
成员登录名：yuhongming-2026
角色：Implementer
本会话唯一目标：立项并实现 Q006（真·跨行预取 和/或 减逐行 scalar），在不破坏精度的前提下压低可见点时延；先台账再改代码。

仓库：D:\sageAttention2 算子开发任务书
协作远端：https://github.com/hongwei-2026/cann-ascendc-ops-collab
座位代码：people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/
台账：tasks/active/QuantUpdateScatter_2330/

## 开工前必读（按序）

1. AGENTS.md、docs/collaboration/05-ai-work-protocol.md、02-perf-gate.md
2. people/CLAIM.md（确认 yuhongming-2026）
3. tasks/active/QuantUpdateScatter_2330/STATUS.md
4. tasks/active/QuantUpdateScatter_2330/path-ledger.md（含废弃清单）
5. tasks/active/QuantUpdateScatter_2330/AI-HANDOFF.md
6. tasks/active/QuantUpdateScatter_2330/decisions/DEC_Q005-profile-first.md
7. explorations/EXP_Q005-profile.md + logs/EXP_Q005/*-medians.json
8. 启用 cann-spmv-ascendc skill

## 现行基线（勿弄错）

- 生产代码 = **Q002**（TQue 深度必须为 **2**，与 InitBuffer(...,2) 一致；Codex 曾两次错改成 1，审查时务必打回）
- Q003（收紧 UB/更大 tile）平台无收益 → **已废弃并回退**；禁止重开
- 精度链不可改：fp32 → int16(CAST_RINT) → half → int8；禁止改成 fp32→half→int8
- SyncAll：拷贝后多核写 out 前仍需要，勿轻易删
- OpDef 必须保持 **4** 签名（fp16/bf16 × int32/int64）；缺签名会 ACL 161002

## Q005 设备结论（冲榜依据）

| 诊断 | out µs | in µs | 含义 |
|------|--------|-------|------|
| copy_heavy | 29.62 | 9.10 | 大异址才有整表拷贝税 ~20µs |
| narrow_rows | 495.73 | 457.42 | 多行更新主导；scalar≈93，MTE/vec 全热；拷贝次要 |
| wide_rows | 41.37 | 40.43 | out≈in；scalar ≳ vec |

平台可见点（Q002）约 5.33/5.74/6.52/5.67/7.83 µs，榜最优约 3.68–4.00 µs。
→ **冲榜不要优先整表拷贝，也不要再抠 UB。**

## Q006 假设（本会话）

当前 Q002 逐 tile `EnQue` 后立刻 `DeQue`，不是真·跨行双缓冲。
假设：多行路径上 MTE2 与 V/MTE3 未叠满，且逐行 scalar 寻址偏贵。
方法（择一或组合，改前写 path-ledger 新行 + 短 DEC）：
A. 真·跨行预取：下一行 updates MTE2 与当前行 quant/V/MTE3 重叠（双槽队列真正错开）
B. 减逐行 scalar：合并地址计算、减少分支、indices 已缓存则避免回退 GetValue
C. 不要做：Q003 式 UB 重算、整表 copy 大改（除非新证据证明平台大量大异址）

止损（02-perf-gate）：
- 时限 1 人日
- 云 `-e` 样例必须 PASS（mismatched=0）
- 平台可见点相对 Q002 增益 <2% 或精度挂 → 回退 Q002，写 DEC
- 禁止假 PASS、删历史台账

## 硬围栏

- 只写：people/yuhongming-2026/QuantUpdateScatter_2330/** 与 tasks/active/QuantUpdateScatter_2330/**
- 禁止：其它题、他人 people/、无 ledger 换路、把 TQue 深度改回 1
- 每次探索写 path-ledger；复杂过程写 explorations/EXP_Q006-*.md
- 云日志下载到 explorations/logs/EXP_Q006/ 并 push main
- 诊断工具：code/examples/profile_q005.sh 、code/tools/README_Q005.md（可复测 narrow/wide）

## 建议工作流

1. git pull --ff-only origin main
2. 读上述文件，用一两句话复述唯一目标与废弃清单
3. 在 path-ledger 登记 Q006 行 + decisions/DEC_Q006-*.md
4. 改 op_kernel/quant_update_scatter.h（必要时 tiling）；保持 4 签名与 RINT 链
5. 本地能跑的 CPU/契约测试先过；再给云测命令：
   cd people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code
   export DEVICE_ID=0
   bash build.sh -j8 -e 2>&1 | tee /tmp/qus2330_q006_build_e.log
6. -e PASS 后上传平台，对比 Q002 可见点；并用 profile_q005.sh 复测 narrow_rows_in / wide_rows_in
7. 更新 STATUS / path-ledger / AI-HANDOFF / dev-log；push

先汇报你的具体改法与风险，再深度编码。不要扩大 scope。
```
