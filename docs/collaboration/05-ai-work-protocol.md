# AI 工作规范流程（防搞乱项目）

适用：本协作仓内一切 AI（Cursor Agent、子 Agent、外部模型）与「人+AI」混编。  
优先级：**任务书 > CONTRIBUTING / 本文件 > 用户口头一句「顺便」> 模型自发发挥**。

目标：AI **加速记账与实现**，禁止 AI **扩 scope、串题、无台账换路、假进度**。

---

## 1. 总流程（AI 必须按门禁走）

```text
[接单] 读 AGENTS.md + 本题 STATUS/TASK/SUMMARY/path-ledger
   ↓
[定界] 填写/核对「会话卡片」：题名、阶段、允许改的路径、禁止事项
   ↓
[阶段门] 未过门禁 → 只做门禁允许的事（见 §3）
   ↓
[记账] 新道路先写 path-ledger 一行（假设/时限）再动手
   ↓
[执行] 最小改动；只碰允许路径；一事一结
   ↓
[收尾] 更新 path-ledger 结论 + dev-log；STATUS 若变化则改；不擅自开新题
```

**没有会话卡片 / 没有明确题名 → AI 默认只读，不改文件。**

会话卡片模板：[docs/templates/ai-session-card.md](../templates/ai-session-card.md)

---

## 2. 角色隔离（多 AI 同时开时）

| 角色 | 可写路径 | 禁止 |
|------|----------|------|
| Analyst（分析） | 仅 `tasks/.../<本题>/analysis/AI-*.md` | 改 Kernel/Host、改其他题、改全局规范（除非专项） |
| Summarizer（汇总） | 仅本题 `analysis/SUMMARY.md` + 回写本题 `path-ledger` 相关行 | 开始编码 |
| Designer | 本题 `design.md`、`TASK.md` 澄清项 | 大段实现代码 |
| Implementer（实现） | 题目约定的代码目录 / fork 仓；本题 `dev-log`、`path-ledger` | 改 `docs/collaboration` 规范（除非 Owner 授权）；动其他 `tasks/*` |
| Librarian（规范） | `docs/`、`CONTRIBUTING`、`AGENTS.md` | 改某题实现细节冒充规范 |

硬规则：

1. **同一时刻、同一题，只允许 1 个 Implementer 写代码。** 多个分析 AI 可并行，但只写各自 `AI-n_*.md`。  
2. 子 Agent 产出必须落在**本题**约定路径；写到题根或仓库根的文件，主 Agent **必须迁回**或删除。  
3. 禁止用「同一对话」假装三路分析凑数（见 [04-multi-ai-analysis.md](04-multi-ai-analysis.md)）。

---

## 3. 阶段门禁（AI 擅自跨越 = 违规）

| 当前 STATUS / 阶段 | AI 允许 | AI 禁止 |
|--------------------|---------|---------|
| 规范就绪 / 材料未齐 | 同步材料到本题 `materials/`、登记 TASK | 编码、性能猜测当结论 |
| 分析中 | 只写 `analysis/AI-*` | 改设计当实现、改其他题 |
| 分析定稿 | 写/改 `design.md`、澄清 TASK | 无 Owner 确认的深度 Kernel |
| 设计中 | 设计文档、用例表 | 性能极限优化 |
| 实现中 | 按 SUMMARY P0 最小实现 | 跳过台账开 P2；同时开互斥大改 |
| 性能中 | 有界优化 + DEC | 小 shape 死磕；废弃清单重开 |
| 归档 | 只读；教训摘进 `path-ledger` 摘要 | 「顺便重构」归档题 |

判断依据以本题 `STATUS.md` 为准；口嗨「先写着」不能覆盖 STATUS。

---

## 4. 目录与文件围栏（防搞乱）

### 4.1 默认允许（单题会话）

```text
tasks/active/<本题>/          ✅ 本题文档与台账
tasks/active/<本题>/analysis/ ✅ 分析
tasks/active/<本题>/materials/✅ 只读材料（改模板副本须声明）
```

### 4.2 默认禁止（除非会话卡片写明）

```text
tasks/active/<其他题>/     ❌
tasks/archive/**           ❌ 除非归档摘要专项
skills/                    ❌ 只读参考；改 skill 另开 Librarian 会话
规范/                      ❌ 任务书原文只读
仓库根乱建 *.md / 脚本      ❌
删除 path-ledger / SUMMARY / 历史分析  ❌
```

### 4.3 协作仓 vs 合入仓

- **本仓**（`cann-ascendc-ops-collab`）：规范、台账、分析、小模板；**不堆**大型算子实现主战场。  
- **合入 fork 仓**：Host/Kernel 实现。AI 改代码前必须在会话卡片写清 **repo 路径 + 分支**。  
- 禁止把 A 题 fork 的改动提交进 B 题目录「备份」。

---

## 5. 行为禁令（红线）

1. **禁止串题**：不得把 SpMV / SageAttention / CUBE 的门禁数字、目录、结论直接当成本题事实。  
2. **禁止无台账换路**：新优化思路未写入 `path-ledger` 不得开工。  
3. **禁止假进度**：未跑通不得写 PASS；不得删 FAIL 日志冒充通过。  
4. **禁止范围膨胀**：用户未点名的「重构全仓 / 顺手再做一题 / 升级规范大扫除」一律不做。  
5. **禁止静默大删**：不得批量删除他人分析、台账行、材料；清理须 Owner 确认并留痕。  
6. **禁止并行互斥实现**：同一题不得同时维护两套 Kernel 主路径（实验须 `exp/` 或独立分支 + 时限）。  
7. **禁止忽略止损**：触碰 [02-perf-gate.md](02-perf-gate.md) 条件必须停并写 `decisions/DECxxx.md`。  
8. **禁止污染工作区**：编译产物、ZIP 原件、profiling 大目录遵守 `.gitignore`；不得强制 add 密钥。

---

## 6. 单次 AI 会话工作法（防空转）

### 开始前（必做）

1. 读 `AGENTS.md`  
2. 读本题 `STATUS.md` → 确认阶段  
3. 若阶段 ≥ 设计：读 `analysis/SUMMARY.md` 废弃清单  
4. 搜本题 `path-ledger`：避免重复已废弃道路  
5. 写出本会话 **唯一目标**（一句话）与 **允许路径**

### 进行中

- 一次会话只推进 **一个** 台账 ID（或明确的一小撮文档）  
- 改代码：最小 diff；不做无关格式化全文件  
- 发现要改围栏外文件 → **先停下来问人**，不得擅自扩大

### 结束后（必做）

- `path-ledger` 结案或标注进行中 + 下一步  
- `dev-log` 追加：做了 / 结果 / 卡住 / 下一步  
- 若产生决策：`decisions/DECxxx.md`  
- 向用户汇报：**改了哪些路径、没改哪些、下一闸是什么**

---

## 7. 人如何指挥 AI（推荐口令）

好的开场：

```text
题：PReluGradUpdate_2329
阶段：按 STATUS
本会话目标：只写 design.md，不改 Kernel
允许路径：tasks/active/PReluGradUpdate_2329/**
禁止：其他 tasks、性能专题、提交合入仓
```

坏的开场（AI 应拒绝或收窄）：

```text
把仓库优化一下 / 顺便看看别的题 / 性能调到最好 / 全部重写
```

Owner 可随时说 **「停，写 DEC」** —— AI 必须停止改代码，只产出决策文档。

---

## 8. 失控恢复（AI 已搞乱时）

| 现象 | 处理 |
|------|------|
| 文件写到错误题目/根目录 | 迁入正确 `analysis/` 或删除；dev-log 记一笔 |
| 未过 SUMMARY 已大改 Kernel | 冻结代码；补分析/SUMMARY；多余优化分支废弃 |
| 同时改多题 | `git status` 按题拆开；只保留授权题 diff |
| 台账与代码不一致 | 以代码事实回写台账，标「补记」；禁止倒改历史结论假装早知 |

恢复完成后才能继续新道路。

---

## 9. 与现有文档关系

| 文档 | 管什么 |
|------|--------|
| [00-principles.md](00-principles.md) | 最简优先等人规 |
| [01-path-ledger.md](01-path-ledger.md) | 台账格式 |
| [02-perf-gate.md](02-perf-gate.md) | 性能止损 |
| [04-multi-ai-analysis.md](04-multi-ai-analysis.md) | 开题 ≥3 分析 |
| **本文件** | **AI 怎么动项目才不乱** |
| 根目录 `AGENTS.md` | Agent 启动短规（本文件摘要） |
