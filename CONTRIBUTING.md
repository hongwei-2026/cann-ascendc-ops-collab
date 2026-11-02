# 团队协作规范

适用：CANN / AscendC 社区任务与开源仓算子工程。  
优先级：**任务书条文 > 本仓规范 > 个人习惯**。

---

## 1. 角色与职责

| 角色 | 职责 |
|------|------|
| Owner | 题目负责人：台账完整、门禁决策、对外 MR/IT |
| Contributor | 实现与记录：每条道路必须落盘 |
| Reviewer | 查台账是否重复、方案是否「最简优先」、止损是否触发 |

同一题目同一时刻只允许 **1 个 Owner**；换人须在 `STATUS.md` 写明交接日期与未闭环道路。

### 1.1 三人并行（people 座位）

精度题耗时长时，允许最多 **3 人** 各写各的实现，互不覆盖：

| 路径 | 用途 |
|------|------|
| `people/<GitHub登录名>/` | 个人代码版本与笔记（**只改自己的**） |
| `people/shared-lessons/` | AI/团队提炼的可复用规律（全员只读引用；写入走 Review） |
| `tasks/active/<题>/` | 共享 STATUS、design、材料、正式 EXP 台账 |

认领与邀请：见 [people/CLAIM.md](people/CLAIM.md)、[people/README.md](people/README.md)。  
可参考另外两人目录与 `shared-lessons/`，但合并别人代码须显式复制到自己目录或开 PR，禁止直接改他人文件夹。

---

## 2. 开发阶段（禁止跳阶段）

```text
0 认领/抄齐任务书 + 同步本题材料（ZIP 等只进本题目录）
0.5 ≥3 路独立 AI/人优化分析 → SUMMARY   ← 见 docs/collaboration/04-multi-ai-analysis.md
1 设计（design.md，可评审）     ← 不夹带实现代码
2 最简正确实现（功能 + 精度）
3 自测全场景
4 性能对照与（必要时）有界优化
5 交付五件套 + IT / 合入
```

**硬规则**

- **题目隔离**：每题只用 `tasks/.../<题名>/` 下的分析、台账、材料；禁止串题结论、禁止根目录堆杂。  
- **开题分析**：未凑齐 ≥3 路分析并完成 `analysis/SUMMARY.md` 前，不进入深度编码/性能专题。  
- 未完成阶段 2（功能精度基线）**禁止**进入深度性能优化。  
- 阶段 4 每次换路线必须走 [性能止损门禁](docs/collaboration/02-perf-gate.md)。  
- 禁止把「未跑通」写成「已通过」。

---

## 3. 道路台账与探索文档（公开仓防重复造轮子）

本仓 **公开**：后人会直接读文档。算子开发中**每一次探索**（成功或失败）都必须写入本题目录，禁止只留在聊天/私盘。

每个题目目录必须有 `path-ledger.md`；复杂探索另写 `explorations/EXPxxx.md`。

动手前：

1. 全文检索本仓 `path-ledger` / `explorations` / `records`，确认无人试过同思路  
2. 新开一行：假设 → 方法 → 时限 → 预期指标  
3. 结束后补：实测结果 → 结论（采纳 / 废弃 / 挂起）→ 耗时；废弃须写清「别再试的原因」  
4. 推送到 `main`，保证公开可见

**未记账的尝试 = 对公众与团队均未发生过**；重复劳动由未记账方负责。

模板：[docs/templates/path-record.md](docs/templates/path-record.md) · [docs/templates/exploration.md](docs/templates/exploration.md)  
规范：[docs/collaboration/01-path-ledger.md](docs/collaboration/01-path-ledger.md)

---

## 4. 最简优化到达最优效果

「最优」优先定义为 **验收达标**，不是论文级极限性能。

推荐顺序（默认）：

1. 对齐 TBE/标杆语义的正确 Kernel（单核或朴素多核均可）  
2. 全核参与 + 合理 tiling / 双缓冲等**标准套路**  
3. 仅对**未达门禁**的 shape 做专项优化  
4. 小 shape（如 <10us、与 TBE 差 ≤3us）优先走任务书允许的**仿真图 + 分析结论**，不无限抠微秒

禁止：无 profile 证据就改算法；同时开多条互斥大改。

---

## 5. 分支与 PR（本协作仓）

| 分支 | 用途 |
|------|------|
| `main` | 规范、台账、已评审文档 |
| `task/<题名>` | 单题文档与记录更新 |
| `docs/*` | 纯规范修订（可直接短 PR） |

约定：

- 规范变更走 PR，至少 1 人 Review  
- 题目台账更新可同 PR；**不要**在本仓堆大型算子源码（源码在 fork 合入仓）  
- Commit 信息写清「哪一题 / 哪条道路 / 结论」

算子代码仓（ops-nn / ops-math 等）仍按上游 CLA、`/compile`、单 commit 等要求；本仓只管协作与记忆。

---

## 6. 沟通与同步

- 每日/每轮：在题目 `dev-log.md` 追加 3～5 行（做了什么、卡在哪、下一步）  
- 卡住超过止损时限：在题目下开 `decision-*.md`，@ Owner + Reviewer  
- 聊天结论 **24h 内**回写台账，否则视为无效

---

## 7. 题目分类

所有题目登记在 [tasks/README.md](tasks/README.md)：

- `active/`：进行中  
- `archive/`：已结束或长期挂起（含未验收，保留教训）

新题：复制 `docs/templates/task-scaffold/` 到 `tasks/active/<ID>_<短名>/`。

---

## 8. AI 工作规范（防搞乱）

凡使用 AI 助手 / 多 Agent，必须遵守：

- 根目录 [AGENTS.md](AGENTS.md)（每次会话启动短规）  
- [docs/collaboration/05-ai-work-protocol.md](docs/collaboration/05-ai-work-protocol.md)（完整流程、角色隔离、围栏、失控恢复）  
- 开干前复制 [docs/templates/ai-session-card.md](docs/templates/ai-session-card.md) 定界  

摘要：

1. **无题名 → 只读**；有题名 → 只动该题目录。  
2. **按 STATUS 阶段门禁**，禁止分析未齐就编码、禁止无台账换路。  
3. **同时只允许 1 个 Implementer 写代码**；分析 AI 只写 `analysis/AI-*.md`。  
4. 子 Agent 文件写错位置必须迁回约定目录。  
5. 触止损说「停，写 DEC」—— AI 必须停手只出决策文档。

Cursor 已配置 always-apply 规则：`.cursor/rules/ai-work-protocol.mdc`。

---

## 9. 检查清单（开干前）

- [ ] 任务编号、合入仓、硬件、CANN、精度链接、性能门禁已抄入 `TASK.md`  
- [ ] 本题材料只在 `tasks/.../<题>/materials/`，未污染其他题  
- [ ] `analysis/` 已有 ≥3 路独立分析 + 已定稿 `SUMMARY.md`  
- [ ] `path-ledger.md` 已有「基线最简方案」行（与 SUMMARY 一致）  
- [ ] 已知是否允许小 shape 用仿真豁免  
- [ ] 已读本仓 `skills/` 与**本题**归档教训（勿直接抄其他题门禁数字）  
- [ ] （若用 AI）已贴会话卡片：题名 / 唯一目标 / 允许路径 / 禁止项
