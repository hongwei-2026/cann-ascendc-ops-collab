# 道路台账 — PReluGradUpdate_2329

> 开新路前先搜本表与 `tasks/archive/**/path-ledger.md`。  
> 规范：`docs/collaboration/01-path-ledger.md` · 性能止损：`docs/collaboration/02-perf-gate.md`

| ID | 日期 | 作者 | 阶段 | 假设 | 方法 | 时限 | 结果 | 结论 | 耗时 |
|----|------|------|------|------|------|------|------|------|------|
| P001 | 2026-03-22 | team | 协作 | 先立规范与分类可降低后期重复找路成本 | 建 GitHub 协作仓：原则/台账/止损/题目卡；**不开发算子** | 本迭代 | 仓与规范落地 | 采纳 | — |
| P002 | 2026-03-22 | team | 准备 | 官方模板含 dtype/工程骨架，先同步再设计 | 自 `D:\华为算子比赛\` 同步 ZIP → `materials/` 并解压阅读 | 0.5d | Kernel/Tiling TODO；语义见 reference.py | 采纳 | — |
| P003 | 2026-03-22 | team | 分析 | 开题前 ≥3 AI 分析可降低错误优化方向 | AI-1/2/3 独立成文 → `analysis/SUMMARY.md` 定稿；主路径=行片切分+单趟融合+bf16独立key | 本迭代 | 三路齐；裁决见 SUMMARY | 采纳 | — |
| P004 | — | — | 设计 | 按 SUMMARY P0 做最简正确实现可达功能+预期全核带宽 | design.md：InferShape；TilingData(C/inner/ROW_SLICE)；四队列；逻辑GT；MODE_2 bf16 | 设计评审前 | （待做） | 规划中 | — |
| P005 | 2026-03-22 | team | 协作 | AI 无围栏会串题/跳阶段/搞乱目录 | 落地 `AGENTS.md` + `05-ai-work-protocol` + Cursor rule + 会话卡片 | 本迭代 | 规范已合入协作仓 | 采纳 | — |
| P006 | 2026-11-02 | team | 分析 | 本地 Codex/Claude 设计比 Cursor 粗案更细 | 导入 `PReluGradUpdate_problem_2329_plan`；SUMMARY 以 design_final 为实施主依据（Select/Inf/模式R） | 本迭代 | 已合并定稿 | 采纳 | — |
| P007 | 2026-11-02 | team | 实现 | 用 git 同步 CodeArts，避免网页上传 | `workdir/code` + `SYNC_CODEARTS.md`；云端 clone/pull+rsync | 进行中 | 命令已写 | 采纳 | — |
| P008 | 2026-11-02 | team | 功能 | 空 Kernel 必 WA，作基线 | 云端 `build.sh -e` | 0.5h | dx 192/192、update 95/192 FAILED | 废弃(空实现) | — |
| P009 | 2026-11-02 | team | 功能 | 按 design_final 实现可过样例 | InferShape+Tiling+Compare/Select Kernel；key 0/1/2 | 本迭代 | 首次云编因 bf16 Cast 失败 | 进行中 | — |
| P010 | 2026-11-02 | team | 环境 | bf16 用 Cast 升精度 | `Cast(bfloat16_t↔float)` | 0.5h | `not support bf16 type cast` → FATBIN 缺 .o | **废弃** | — |
| P011 | 2026-11-02 | team | 功能 | bf16 避开 Cast | 位扩展/RNE 截断逐元素；half/float 仍 Cast | 本迭代 | 编过；样例仍 WA ~97/192 | 采纳(编译) | — |
| P012 | 2026-11-02 | team | 功能 | WA≈半错=Select 极性反 | 交换 Select src0/src1（禁改 LE） | 本迭代 | 换后仍 95/192；见 EXP004 | **废弃(非根因)** | — |
| P013 | 2026-11-02 | team | 协作 | 云日志须落盘公开仓 | 本机 tar→`explorations/logs/EXP20261102_161116`→push | 0.5h | build OK + WA 95/192 原文已归档 | 采纳 | — |
| P014 | 2026-11-02 | team | 功能 | mask 失效→恒选 src1 | 解压日志 + seed42 复算错数；写 EXP004 | 0.5h | 95≡恒正支；97≡恒负支；对上 EXP003 前后 | **采纳(根因)** | — |
| P015 | 2026-11-02 | team | 功能 | 修 Compare/Select mask 通路 | 恢复 Select(g,wg)/(0,xg)；CompareScalar+PipeBarrier；mask 按 256-bit 对齐 | 本迭代 | 已改 workdir；待云端 `-e` | 进行中 | — |

## 性能道路预声明（未到阶段勿开）

| ID | 触发条件 | 默认策略 |
|----|----------|----------|
| P1xx | 功能精度基线采纳后，全核 case < 95% TBE | 标准 tiling/全核等有界优化；触止损写 DEC |
| P1yy | 小 shape <10us 且差 ≤3us | **仿真图 + 分析**，默认不抠微秒 |
