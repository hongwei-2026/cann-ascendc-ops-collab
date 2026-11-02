# CANN AscendC 算子团队协作仓

团队长期开发用的**规范 + 题目台账 + 道路记录**仓库。  
目标：少走弯路、避免重复试错、用最简路径先达验收门槛。

> **本阶段不做算子实现。** 先立规矩、分类题目、固定记录方式，再按门禁开干。

## 仓库结构

```text
.
├── CONTRIBUTING.md              # 协作总规（必读）
├── docs/
│   ├── collaboration/           # 原则、性能止损、道路台账、评审清单
│   └── templates/               # 可复制模板
├── tasks/
│   ├── README.md                # 题目分类索引
│   ├── active/                  # 进行中
│   └── archive/                 # 历史/未验收归档
├── skills/                      # AscendC 社区任务 skill 参考副本
├── 规范/                        # 任务书原文归档
└── records/                     # 跨任务操作记录摘要
```

## 核心原则（一句话）

1. **最简优先**：先做能过功能/精度的最简正确实现，再谈性能。  
2. **开题先分析**：每题 ≥3 路独立 AI/人分析，再 `SUMMARY`，然后才设计/编码。  
3. **先记账再改路**：换优化路线前必须查台账、写假设、设时限。  
4. **止损硬门禁**：超时/无增益 → 停、写结论、团队决策；禁止闷头死磕。  
5. **一题一目录隔离**：分析/材料/台账只进该题目录，禁止串题。

## 当前活跃题目

| 题目 | 目录 | 状态 |
|------|------|------|
| PReluGradUpdate（problem 2329） | [tasks/active/PReluGradUpdate_2329](tasks/active/PReluGradUpdate_2329) | 规范就绪 / 未开发 |

历史题目见 [tasks/README.md](tasks/README.md)。

## 新人上手

1. 读 [CONTRIBUTING.md](CONTRIBUTING.md)  
2. 读 [docs/collaboration/00-principles.md](docs/collaboration/00-principles.md)  
3. 接题时复制 `docs/templates/`，在 `tasks/active/<题名>/` 落盘  
4. 动手前先写 `path-ledger.md` 的「基线方案」一行

## 上游与 skill

- 社区任务入口：https://gitcode.com/cann/cann-ops-competitions  
- 本机/参考 skill：`skills/`（源自 `cann-spmv-ascendc`）
