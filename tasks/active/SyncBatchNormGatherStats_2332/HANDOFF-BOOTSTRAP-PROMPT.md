# SyncBatchNormGatherStats_2332 — 新开窗开题提示词

把下面「整段粘贴」框复制到**新对话**。  
本题此前未入库；先脚手架 + 分析，再实现。ZIP **不要整包提交 build/**。

---

```text
【新对话 · SyncBatchNormGatherStats_2332 · 开题脚手架+分析 · 勿串其它题】

题名：SyncBatchNormGatherStats_2332
成员登录名：yuhongming-2026
角色：Implementer
本会话唯一目标：按本仓协作规范完成本题开题——导入 ZIP 基线（剔 build）、建 tasks/people 台账、≥3 路独立分析 + SUMMARY 定稿；未得用户确认前不做深度 Kernel/性能死磕。

仓库根：D:\sageAttention2 算子开发任务书
协作远端：https://github.com/hongwei-2026/cann-ascendc-ops-collab
来源 ZIP（不入库原包）：D:\华为算子比赛\SyncBatchNormGatherStats_2332.zip
座位代码目标：people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/
共享台账：tasks/active/SyncBatchNormGatherStats_2332/

## 0. 启动必读（按序，违反围栏=失败）

1. 根目录 AGENTS.md
2. docs/collaboration/05-ai-work-protocol.md
3. docs/collaboration/02-perf-gate.md（性能阶段才用；现在先别开性能专题）
4. people/CLAIM.md、people/README.md（确认座位 yuhongming-2026）
5. 启用 skill：cann-spmv-ascendc（C:\Users\yuhon\.cursor\skills\cann-spmv-ascendc\SKILL.md）

## 1. 硬围栏

- 只写：
  - people/yuhongming-2026/SyncBatchNormGatherStats_2332/**
  - tasks/active/SyncBatchNormGatherStats_2332/**
- 禁止：其它题、他人 people/、仓库根乱丢文件、改 skills/、无授权「顺便优化全仓」
- 禁止串题（尤其 QuantUpdateScatter_2330 / ConfusionTransposeD / PRelu 等）
- 无 SUMMARY 定稿 → 禁止深度编码 / 性能专题
- 无 path-ledger 新行 → 禁止换优化路线
- 禁止假 PASS；失败探索也必须落盘
- 云日志（含编译失败全文）归档到本题 explorations/logs/ 并 git push main
- 子 Agent 产物只能进 analysis/ 或 people/yuhongming-2026/... 约定目录

## 2. 本题已知（来自 ZIP，需复核）

算子：SyncBatchNormGatherStats（aclnn 名见 autogen/包内头文件）。

输入：total_sum、total_square_sum、sample_count、mean、variance；属性 momentum、eps。  
输出：batch_mean、batch_invstd、new_mean、new_variance。

OpDef 已见 4 组 dtype 配对（fp32/fp16 × sample_count int32/int64）；new_mean/new_variance 在 def 里声明为 float——以源码为准，勿臆造。

**精度铁律（reference.py 文件头，与 910B 内置对齐）**：
- 浮点输入先按 **fp16 量化**再算（fp32 用例也会被 Cast 进同一套 fp16 核语义）。
- 中间用 fp32：batch_mean / batch_var / batch_invstd / EMA 更新。
- 输出再按 fp16 网格写出（即便声明 dtype 是 fp32，值也在 fp16 网格上）。
- count==1 时 count/(count-1)=inf，与设备一致；勿「修掉」成有限值。
- 权威语义：workdir/code/samples/reference.py（导入后以座位内副本为准）。

ZIP 内含可编译工程 + build/ 产物；**入库时剔除 build/、__pycache__、大体积中间物**，只保留源码与必要 samples。

## 3. 本会话交付清单（按序做）

A. git status / pull 了解 main；确认不覆盖他人未提交工作。  
B. 建台账骨架：
   - tasks/active/SyncBatchNormGatherStats_2332/{STATUS.md,path-ledger.md,dev-log.md,AI-HANDOFF.md,CLOUD-RUN.md}
   - STATUS 初始阶段写「分析中」或「实现中（基线导入）」并一句话说清下一闸
C. 导入 ZIP → people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/（无 build）。  
D. ≥3 路独立分析 → analysis/AI-1.md、AI-2.md、AI-3.md（或等价命名），再写 analysis/SUMMARY.md（含废弃清单 / P0）。  
E. 未得用户「可以编码」确认前：最多做只读摸底、参考对拍脚本、脚手架；不要开性能专题。  
F. 每步更新 path-ledger + dev-log；需要公开时 commit 并 push main（用户要求才 commit 时按其指示；协议要求日志可见则提醒 push）。

## 4. 云测（有代码且用户授权后）

典型路径（以实际 CLOUD-RUN.md 为准）：
  cd /mnt/workspace/cann-ascendc-ops-collab
  git pull --ff-only origin main
  cd people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code
  export DEVICE_ID=0
  bash build.sh -j8 -e 2>&1 | tee /tmp/sbngs2332_build_e.log
样例 PASS ≠ 平台 50 点 PASS；未下载全文 log 不得只留截图。

## 5. 开工动作

1. 一句话复述：题名、座位、本会话唯一目标、允许改动路径。  
2. 检查 ZIP 是否存在；列出将创建的目录树计划。  
3. 先脚手架+分析，汇报 SUMMARY 要点与下一闸，再等用户下令深度实现。

不要并行开其它题。不要把 QuantUpdateScatter 的优化惯性套过来（本题是 BN stats gather，精度模型不同）。
```
