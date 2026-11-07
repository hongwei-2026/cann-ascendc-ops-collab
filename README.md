# CANN AscendC 算子三人协作仓

**仓库**：https://github.com/hongwei-2026/cann-ascendc-ops-collab（公开）

精度题往往要扛很久：三人**各写各的实现目录**，互不踩盘；同时用共享台账 + `shared-lessons` 互相参考，少重复踩坑。

---

## 成员与座位

| 座位 | GitHub | 个人目录 | 状态 |
|------|--------|----------|------|
| 1 · Owner | [hongwei-2026](https://github.com/hongwei-2026) | [`people/hongwei-2026/`](people/hongwei-2026/) | 已认领 |
| 2 | [yuhongming-2026](https://github.com/yuhongming-2026) | [`people/yuhongming-2026/`](people/yuhongming-2026/) | 已预留 · 请 Accept 邀请 |
| 3 | [JadeYara](https://github.com/JadeYara) | [`people/JadeYara/`](people/JadeYara/) | 已预留 · 请 Accept 邀请 |

认领表详情：[people/CLAIM.md](people/CLAIM.md) · 隔离规则：[people/README.md](people/README.md)

受邀同学：打开 GitHub 通知 / 邮件，对 `cann-ascendc-ops-collab` 点 **Accept invitation**，再 `git clone` 本仓。

---

## 怎么分工（一句话）

| 放哪里 | 写什么 | 谁能动 |
|--------|--------|--------|
| `people/<你的登录名>/` | **你的** Kernel / Host / 个人笔记与试验版 | **仅本人** |
| `people/shared-lessons/` | AI/团队提炼的规律、坑（短文） | 提案后 Review 合入；全员可读 |
| `tasks/active/<题名>/` | STATUS、design、材料、正式 EXP 台账 | 全员可补台账；**不要**三人抢同一 workdir |

可以**只读**另外两人目录和 `shared-lessons/`；要吸收别人代码 → **复制进自己目录**或开 PR，禁止直接改他人文件夹。

---

## 当前题目

| 题目 | 共享台账 | 实现（按人） |
|------|----------|--------------|
| **PReluGradUpdate**（2329） | [tasks/active/PReluGradUpdate_2329](tasks/active/PReluGradUpdate_2329) | [hongwei workdir](people/hongwei-2026/PReluGradUpdate_2329/) · 另两人自建 `people/<你>/PReluGradUpdate_2329/workdir/` |
| **MultilabelMarginLoss**（2327） | [tasks/active/MultilabelMarginLoss_2327](tasks/active/MultilabelMarginLoss_2327) | [yuhongming workdir](people/yuhongming-2026/MultilabelMarginLoss_2327/) |
| **LayerNormV3**（2326） | [tasks/active/LayerNormV3_2326](tasks/active/LayerNormV3_2326) | [yuhongming workdir](people/yuhongming-2026/LayerNormV3_2326/) |

PRelu：实现中（样例精度攻坚；共识坑见 [shared-lessons · PRelu](people/shared-lessons/PReluGradUpdate_2329/)）。  
MultilabelMarginLoss：性能中。2026-11-07 座位对齐按需 workspace，语义自测 PASS；这次未跑设备 `-e`。  
LayerNormV3：实现中（yuhongming 座位；语义自测已过，910B `-e` 未跑）。

CodeArts 同步：只覆盖**你自己的** `people/<你>/.../workdir/code/` → 题包 `code/`（见题内 `SYNC_CODEARTS.md`）。

---

## 新人 5 步

1. Accept 仓库邀请 → `git clone https://github.com/hongwei-2026/cann-ascendc-ops-collab.git`  
2. 读 [CONTRIBUTING.md](CONTRIBUTING.md)、[people/README.md](people/README.md)、[people/shared-lessons/](people/shared-lessons/)  
3. 在自己的 `people/<登录名>/` 建 `<题名>/workdir/`（可从 hongwei 目录**复制**后改）  
4. 用 AI 前读 [AGENTS.md](AGENTS.md)，会话里写明：**题名 + 你的 GitHub 登录名**  
5. 每次探索写入题内 `path-ledger` / `explorations/`，并 **push `main`**

---

## 仓库树

```text
.
├── AGENTS.md                 # AI 必读短规
├── CONTRIBUTING.md           # 协作总规
├── people/                   # 三人座位 + shared-lessons
├── tasks/active|archive/     # 按题共享台账与材料
├── docs/collaboration/       # 原则 / 止损 / 多路分析 / AI 流程
├── skills/                   # AscendC 社区任务 skill 参考
├── 规范/                     # 任务书原文归档
└── records/                  # 跨任务摘要
```

---

## 核心原则

1. **最简优先**：先过功能/精度，再谈性能。  
2. **开题先分析**：≥3 路独立分析 → `SUMMARY` → 再深度编码。  
3. **先记账再换路**：`path-ledger` 不写 = 对团队等于没试过。  
4. **失败也落盘**：EXP + 云日志归档；禁止只截图。  
5. **止损硬门禁**：超时/无增益就停，写结论。  
6. **座位隔离**：代码只进自己的 `people/<你>/`；台账进 `tasks/`。  
7. **AI 有围栏**：无题名 / 无登录名 → 只读（详见 AGENTS.md）。

---

## 上游

- 社区任务：https://gitcode.com/cann/cann-ops-competitions  
- 本机 skill 参考：`skills/`
