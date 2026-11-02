# 三人并行开发区（people）

本仓最多 **3 名实现者** 并行同一题：每人只写自己的目录，互不覆盖；可 **只读** 参考另外两人与 `shared-lessons/`。

## 布局

```text
people/
├── README.md                 # 本说明
├── CLAIM.md                  # 座位认领表（谁占哪格）
├── shared-lessons/           # 【总文件夹】AI 提炼、全员可引的规律/坑
│   └── ...
├── hongwei-2026/             # 座位1 Owner
├── yuhongming-2026/          # 座位2 已预留
└── JadeYara/                 # 座位3 已预留
```

## 规则（硬）

1. **只改自己的 `people/<你的登录名>/`**；禁止直接改别人目录里的代码。  
2. **可以读** 另外两人目录 + `shared-lessons/` + `tasks/active/<题>/` 共享台账。  
3. **不要**在仓库根、`tasks/*/workdir` 堆个人实验（`tasks/.../workdir` 若仍存在，视为历史/指针，新工作进 `people/<你>/`）。  
4. 想把某条经验升成「全员共识」→ 写成 PR，放进 `shared-lessons/`（见该目录 README）；需至少一名其他成员或 Owner 点头。  
5. 同一人目录内可有多题子目录；一题可有多个 `v1`/`v2` 实验子目录，但 **对外同步 CodeArts 时写清用哪一版**。

## 新人三步

1. 读 [CLAIM.md](CLAIM.md)，认领空座位（改名 `seat-3` → 你的 GitHub 登录名，或占用已预留目录并改 CLAIM 状态）。  
2. 复制 `people/_template/` 到你的目录（若还没有）。  
3. 在本题下建 `workdir/`，从 `tasks/active/<题>/` 的材料/设计只读起步；自己的 Kernel 只进自己的 workdir。
