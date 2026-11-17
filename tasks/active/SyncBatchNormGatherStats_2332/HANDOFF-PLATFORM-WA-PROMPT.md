# SyncBatchNormGatherStats_2332 — 平台 WA 排障提示词

把下面「整段粘贴」框复制到**新对话**。本题已开题、云样例 PASS；平台提交 **Wrong Answer**（可见点 1–5 Pass，后续隐藏）。本会话只做精度排障，不开性能。

---

```text
【新对话 · SyncBatchNormGatherStats_2332 · 平台 Wrong Answer 排障 · 勿串其它题】

题名：SyncBatchNormGatherStats_2332
成员登录名：yuhongming-2026
角色：Implementer
本会话唯一目标：定位并修复平台「Wrong Answer」（可见测点 1–5 已 Pass、错误占比 0%；测点 6 起隐藏显示 *）。以最小 diff 对齐 910B/reference 语义；未得用户确认前不做性能优化。

仓库根：D:\sageAttention2 算子开发任务书
协作远端：https://github.com/hongwei-2026/cann-ascendc-ops-collab
座位代码：people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/
共享台账：tasks/active/SyncBatchNormGatherStats_2332/
提交清单：tasks/active/SyncBatchNormGatherStats_2332/deliverables/SUBMIT_LINKS_S002.md

## 0. 启动必读（按序）

1. 根目录 AGENTS.md
2. docs/collaboration/05-ai-work-protocol.md
3. people/CLAIM.md（确认座位 yuhongming-2026）
4. 本题 STATUS.md / path-ledger.md / analysis/SUMMARY.md（含废弃清单）
5. AI-HANDOFF.md
6. 权威语义：workdir/code/samples/reference.py（文件头注释必读）
7. 启用 skill：C:\Users\yuhon\.cursor\skills\cann-spmv-ascendc\SKILL.md

## 1. 硬围栏

- 只写：
  - people/yuhongming-2026/SyncBatchNormGatherStats_2332/**
  - tasks/active/SyncBatchNormGatherStats_2332/**
- 禁止：其它题、他人 people/、仓库根乱丢、无授权全仓优化
- 禁止串题（QuantUpdateScatter / ConfusionTransposeD / PRelu 等）
- 无 path-ledger 新行 → 禁止换优化路线
- 禁止假 PASS；失败也要落盘；云日志全文归档并 push main
- **本会话禁止性能专题**（平台可见点约 5.9–6.2μs vs 最优约 3.7μs 仅作记录，先修 WA）

## 2. 已完成事实（勿重复开题）

| 项 | 事实 |
|----|------|
| 开题 | S001：ZIP 导入 + ≥3 路分析 + SUMMARY 定稿 |
| 云冒烟 | EXP_S002：`build.sh -j8 -e` **PASS**（四输出 mismatched=0/6）；tip 基线 `219e221`；台账已归档 |
| 平台提交 | 按 SUBMIT_LINKS_S002 上传 7 个 Host/Kernel 源文件（无 CMakeLists） |
| 平台结果 | 总评 **Wrong Answer**；测点 1–5 **Pass / 0.00%**；用时约 5.87–6.23μs（最优约 3.7–3.8μs）；测点 6 起界面为 `*`（隐藏，极可能从此挂） |

一句话：前 5 个可见点精度过；**WA 来自隐藏测点**，不是「完全算错」。

## 3. 精度铁律（修 WA 时禁止破坏）

权威：`samples/reference.py`（与 910B 内置实测对齐）：

1. 浮点输入先 **fp16 量化**再算（fp32 用例也会被 Cast 进同一 fp16 核语义）。
2. 中间 **fp32**：batch_mean / batch_var / batch_invstd / EMA。
3. 四输出再落 **fp16 网格**（即便声明 dtype 是 fp32）。
4. `count==1` → `n/(n-1)=+inf`；**勿擅自修成有限值**。
5. `new_mean` / `new_variance` OpDef 恒 **float**；`batch_*` 与输入同 dtype。
6. Test_050 类：`total_square_sum=100.001`→fp16=`100.0` → `batch_var` 恰为 0 → `invstd=1/sqrt(eps)≈10000`；跳过量化会错。

## 4. SUMMARY 已标高危假说（按优先级查）

隐藏点 WA 优先怀疑（有证据再改，一次一条 path-ledger）：

1. **`sample_count` 多元素**：Kernel `SumSampleCount()` **求和**；reference 只取 `reshape(-1)[0]`。若平台语义是「已是全局 n、只读首元」，则求和会错；反之若每卡 count 应加和则现实现可能对。**必须用内置/平台行为裁决，禁止无证据折 countNum=1。**
2. **`world_size>1`**：Kernel 沿 device 维累加；reference `reshape(-1)` 在 W>1 时**不是**同一公式。前 5 点可能都是 W=1。
3. **fp16 路径 `batch_invstd`**：标量补 NaN 之后 `StoreAsInputDtype<half>` 再 `CAST_RINT`，可能**再次吞掉**刚补的 NaN（AI-3 已标）。
4. **`count==1` → new_var ±inf/nan**：判题器对特殊值有规则；勿改成有限。
5. **负 `batch_var+eps` / var+eps==0**：invstd 须 QuietNaN / +inf；向量 Sqrt+Cast 会丢特殊值。
6. **dtype 组合**：OpDef 四组（fp32/fp16 × int32/int64 count）；可见点可能只覆盖子集。
7. **InferShape 跟 mean、Tiling 跟 sum 的 C**：shape 不一致时图过核错（少见但要查）。
8. **尾核 / C 非整块**：`kMinBlock=8`、`ubFactor` 对齐；确认按 `len` 写回而非按 ubFactor 写穿。

**不要先做**：删 fp16 量化、修 inf、为追平台用时改双缓冲/min-block、重写整算子、串题套数字。

## 5. 建议工作法

A. 读 STATUS / SUMMARY 废弃清单 / path-ledger；开新行如 **S003**（假设一句话 + 时限）。
B. 下载平台工程（页面「下载工程」）或对照内置 bin：找出隐藏点与可见点差异（shape/dtype/count/W/eps/momentum）。
C. 本地/云复现：最小用例脚本或扩展 reference；`build.sh -e` 不够；针对假说构造 case。
D. **最小 diff** 修 Kernel/Host；云 `-e` 回归首点；再交平台。
E. 结案：更新 path-ledger / explorations/EXPxxx / logs 全文 / STATUS / push main。

## 6. 关键代码位置

- Kernel：`op_kernel/sync_batch_norm_gather_stats.h`（QuantFp16Vec、world 循环、count 求和、inf、NaN 补丁、Store）
- Host：`op_host/sync_batch_norm_gather_stats_tiling.cpp` / `_def.cpp` / `_infershape.cpp`
- 语义：`samples/reference.py`

## 7. 开工

1. 一句话复述：题名、座位、唯一目标（修平台 WA）、允许路径。
2. 先写 path-ledger S003 假设，再读代码/平台下载包找隐藏点差异。
3. 修完汇报：改了哪些文件、验证了什么、是否建议重交平台；提醒 push。

不要并行开其它题。性能等 WA 全绿后再开（另写台账行）。
```
