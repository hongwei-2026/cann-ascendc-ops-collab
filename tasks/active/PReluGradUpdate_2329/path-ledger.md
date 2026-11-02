# 道路台账 — PReluGradUpdate_2329

> 开新路前先搜本表与 `tasks/archive/**/path-ledger.md`。  
> 规范：`docs/collaboration/01-path-ledger.md` · 性能止损：`docs/collaboration/02-perf-gate.md`

| ID | 日期 | 作者 | 阶段 | 假设 | 方法 | 时限 | 结果 | 结论 | 耗时 |
|----|------|------|------|------|------|------|------|------|------|
| P001 | 2026-03-22 | team | 协作 | 先立规范与分类可降低后期重复找路成本 | 建 GitHub 协作仓：原则/台账/止损/题目卡；**不开发算子** | 本迭代 | 仓与规范落地 | 采纳 | — |
| P002 | — | — | 准备 | 官方模板含 dtype/工程骨架，先同步再设计 | 从 CodeArts `/mnt/workspace` 同步 `PReluGradUpdate_problem_2329_template.zip` 并校验任务书 | 0.5d | （待做） | 规划中 | — |
| P003 | — | — | 设计 | 最简 Ascend C 对齐 TBE 语义可先过功能精度 | Host/Kernel 分离；逻辑值比较；泛化用例表；暂不做极限优化 | 设计评审前 | （待做） | 规划中 | — |

## 性能道路预声明（未到阶段勿开）

| ID | 触发条件 | 默认策略 |
|----|----------|----------|
| P1xx | 功能精度基线采纳后，全核 case < 95% TBE | 标准 tiling/全核等有界优化；触止损写 DEC |
| P1yy | 小 shape <10us 且差 ≤3us | **仿真图 + 分析**，默认不抠微秒 |
