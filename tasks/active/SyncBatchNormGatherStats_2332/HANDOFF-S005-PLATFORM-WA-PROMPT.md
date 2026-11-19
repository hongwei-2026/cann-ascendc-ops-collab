# SyncBatchNormGatherStats_2332 — S005 平台 WA 排障提示词

把下面「整段粘贴」框复制到**新对话**。

---

```text
【新对话 · SyncBatchNormGatherStats_2332 · S005 平台 WA · 勿串题 · 勿重开 S003/S004】

题名：SyncBatchNormGatherStats_2332
成员登录名：yuhongming-2026
角色：Implementer
本会话唯一目标：在 S002 性能基线之上，另找隐藏点 Wrong Answer 根因并最小修复。S003/S004（Cast 后标量保 NaN/Inf）已证明：平台仍 WA，且可见点用时变差 → 已回退，禁止重开。

仓库根：D:\sageAttention2 算子开发任务书
座位代码：people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/
台账：tasks/active/SyncBatchNormGatherStats_2332/

## 0. 启动必读

1. AGENTS.md + docs/collaboration/05-ai-work-protocol.md
2. people/CLAIM.md（座位 yuhongming-2026）
3. 本题 STATUS / path-ledger / analysis/SUMMARY.md（废弃清单）
4. samples/reference.py
5. explorations/EXP_S003.md、EXP_S004.md（失败教训）

## 1. 硬围栏

- 只写 people/yuhongming-2026/SyncBatchNormGatherStats_2332/** 与 tasks/active/SyncBatchNormGatherStats_2332/**
- 禁止串题；无 path-ledger 新行不换路；禁止假 PASS；云 log 归档并 push
- **禁止重开**：QuantFp16Vec / StoreAsInputDtype 里「Cast 后标量扫 NaN/Inf」整族补丁（S003/S004）
- 本会话默认 **先修 WA**；可见点用时约 5.9μs（S002）勿再故意加标量环；性能专题另开

## 2. 已证实事实

| 项 | 事实 |
|----|------|
| 平台 | 共约 50 点；1–5 公开 **Pass / 0%**；6–50 隐藏；总评 **Wrong Answer** |
| S002 基线 | 云 `-e` PASS；平台可见点约 **5.87–6.23μs**；隐藏仍 WA |
| S003 | StoreAsInputDtype 保 NaN；云 PASS；平台仍 WA |
| S004 | QuantFp16Vec 也保 NaN；云 PASS；平台仍 WA；可见点约 **6.05–6.69μs**（**更慢**） |
| 结论 | 隐藏 WA **不是**「仅靠输出路径保 NaN/Inf」能修好；标量保活有性能税 |

代码已回退到 S002 语义的 Quant/Store（无 S003/S004 标量环）。工作 tip 见 STATUS。

## 3. 精度铁律（仍有效）

reference.py：输入 fp16 量化 → 中间 fp32 → 输出 fp16 网格；count==1 → inf；new_* 恒 float 存储。
**勿**为修 WA 把 inf 改有限、或跳过输入量化。

## 4. S005 优先假说（按序，一次一条台账）

1. **下载平台工程**（页面「下载工程」）：对比隐藏 case 与可见 case 的 shape/dtype/count/W/momentum/eps；用事实否决假说。
2. **`sample_count` 语义**：Kernel 对 numel **求和**；reference 取 `[0]`。任务书写 per-device，但平台若已给全局 n 且 countNum>1，求和会错。应用内置/下载包裁决；有证据才改。
3. **`world_size>1`**：Kernel 跨 device 累加；reference reshape 在 W>1 不等价。可见点可能全是 W=1。
4. **dtype 组合**：fp16×int64 等未在 samples 覆盖的路径。
5. **ComputeTile 内已有 invstd 标量 NaN 补丁**：在 QuantFp16Vec 输出前；若隐藏点挂在 new_var 的有限误差（非 NaN），应查公式/ulp，不是再加 Cast 扫环。
6. **尾核 / tiling**：C 非整除、ubFactor>len 写穿等（次优先）。

## 5. 建议工作法

A. path-ledger 开 **S005**（一句话假设）。
B. 优先拿平台下载包或构造 W>1 / 多元素 count / count==1 / fp16 用例，云上复现。
C. 最小 diff；云 `-e` 回归；再交平台。
D. 若必须动特殊值，只允许 **有 profile 证据的、不扫全 tile 热路径** 的写法；禁止恢复 S003/S004 标量全扫。

## 6. 开工

复述题名/座位/目标/允许路径 → 记账 S005 → 先证据后改代码 → 汇报并 push。
```
