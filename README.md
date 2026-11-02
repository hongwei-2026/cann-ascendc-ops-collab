# CANN AscendC 算子团队协作仓

**公开仓库**：https://github.com/hongwei-2026/cann-ascendc-ops-collab  

团队长期开发用的**规范 + 题目台账 + 探索记录**仓库。  
目标：少走弯路、避免重复试错、用最简路径先达验收门槛；**每次探索落盘，后人不再重复造轮子**。

## 仓库结构

```text
.
├── AGENTS.md                    # AI 每次会话必读短规
├── CONTRIBUTING.md              # 协作总规（必读）
├── .cursor/rules/               # Cursor always-apply 围栏
├── docs/
│   ├── collaboration/           # 原则、止损、台账、多路分析、AI 流程
│   └── templates/               # 含 ai-session-card
├── tasks/
│   ├── README.md                # 题目分类索引
│   ├── active/                  # 进行中（一题一目录，严禁串题）
│   └── archive/                 # 历史/未验收归档
├── skills/                      # AscendC 社区任务 skill 参考副本
├── 规范/                        # 任务书原文归档
└── records/                     # 跨任务操作记录摘要
```

## 核心原则（一句话）

1. **最简优先**：先做能过功能/精度的最简正确实现，再谈性能。  
2. **开题先分析**：每题 ≥3 路独立 AI/人分析，再 `SUMMARY`，然后才设计/编码。  
3. **先记账再改路**：换优化路线前必须查台账、写假设、设时限。  
4. **每次探索写文档**：失败也要记进 `path-ledger` / `explorations/`，并 push 公开。  
5. **止损硬门禁**：超时/无增益 → 停、写结论、团队决策；禁止闷头死磕。  
6. **一题一目录隔离**：分析/材料/台账只进该题目录，禁止串题。  
7. **AI 有围栏**：无题名只读；按阶段门禁；一会话一目标；详见 `AGENTS.md`。

## 当前活跃题目

| 题目 | 目录 | 状态 |
|------|------|------|
| PReluGradUpdate（problem 2329） | [tasks/active/PReluGradUpdate_2329](tasks/active/PReluGradUpdate_2329) | 分析定稿 / 未开发 |

历史题目见 [tasks/README.md](tasks/README.md)。

## 新人上手

1. 读 [CONTRIBUTING.md](CONTRIBUTING.md)  
2. 读 [docs/collaboration/00-principles.md](docs/collaboration/00-principles.md)  
3. 用 AI 前读 [AGENTS.md](AGENTS.md) + [05-ai-work-protocol.md](docs/collaboration/05-ai-work-protocol.md)，并填会话卡片  
4. 接题时复制 `docs/templates/task-scaffold/`，在 `tasks/active/<题名>/` 落盘  
5. 动手前先写 `path-ledger.md` 的「基线方案」一行

## 上游与 skill

- 社区任务入口：https://gitcode.com/cann/cann-ops-competitions  
- 本机/参考 skill：`skills/`（源自 `cann-spmv-ascendc`）
