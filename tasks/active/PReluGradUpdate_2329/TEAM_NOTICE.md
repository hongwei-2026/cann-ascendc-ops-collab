# 队友同步 — 现已进入**性能优化**（2026-11-03）

@`yuhongming-2026` @`JadeYara` — 请先读本页再改 Kernel / 交平台。

## 一句话

功能可见点已过（平台 1–5 Pass）。**当前主线是性能**（用时仍 ~ms，参考 ~10μs）。  
默认试交 / 平台链接以 Owner **hongwei `3bd65fb` / P100** 为准。

## 现状

| 项 | 状态 |
|----|------|
| 平台正确性 | **1–5 Pass**（原 case4 `ACL 507035` 已过）；6 隐藏 |
| 平台性能 | ~1.3–2.0 ms vs 参考 ~10–14 μs（约慢两个数量级） |
| 云测 | EXP_P100：三 key Done + 样例 **0/192 PASS**（tip `3bd65fb`） |
| 阶段 | `性能优化中`（见 STATUS） |

## 正确性基线（冻结，勿回退）

1. **P031**：weight `Duplicate` 必须 32B 对齐（未对齐前缀 `SetValue`）——修 case4 507035  
2. Select 用 **count** 重载；**禁止**默认八参 `BinaryRepeatParams`（易平台 Compile Error）  
3. bf16：**禁止** `Cast`；用位转换 + 必要 S_V/S_MTE3  
4. half：可在 `if constexpr(half)` 内 Cast↔fp32；勿 DtypeBridge/`ToFloat` 显式特化（污染各 key）

## 当前性能道路 P100（hongwei，已云测）

代码：`people/hongwei-2026/PReluGradUpdate_2329/workdir/code/`

- `workspace = 0`（去掉无用的 16MB 分配）  
- 小图也拉满多核（公开 fp32 `[4,6,8]` → 24 核）  
- tile 限制在单 channel 内，直接 `Muls`，删除 weight UB 展开  
- **必带补丁**：`Process` 里 `progress += currentNum`（不能 `+= tileLen_`，否则 channel 截断会跳元素 WA）  
- Host：`weights=[C]` 时 `tileLen <= innerSize`

详情：EXP019 · EXP_P100 日志 · path-ledger P100

## 对 yuhongming P032 的审阅

| 点 | 评价 |
|----|------|
| 吸 P031 对齐 `Duplicate` | ✅ 正确，座位应保留 |
| tiling `WS_SYS_SIZE=0` | ✅ 与 P100 一致，有利于砍固定开销 |
| 仍 `BuildWeightVector` + `Mul` | ⚠️ 性能主路径上已被 P100 的 channel-`Muls` 取代；可作对照，勿当默认试交 |
| 仍八参 `Select(..., repeats, BinaryRepeatParams)` | ❌ 平台 Compile Error 高风险；**不要交平台** |
| 小 shape 仍单核整 tile | ⚠️ 与 P100「全核」策略不同；公开点仍易卡在启动/调度 |
| `progress += tileLen_` | 若再叠 channel 截断，需改成 `+= currentNum`（见 P100 补丁） |

结论：P032 **适合座位同步正确性**；**性能默认包仍用 hongwei P100**。若要在 `people/yuhongming-2026/` 试性能，请先吸收 P100 全量语义（含 progress 补丁 + count Select），并新开 `path-ledger` 行，勿直接改 `deliverables/code_best`。

## 性能门禁纪律（`docs/collaboration/02-perf-gate.md`）

- A 类：全核/任务书性能点 → 冲 ≥ TBE 95%（有界道路）  
- B 类：&lt;10μs 且差 ≤3μs → **仿真图+分析**，禁止无限抠  
- 一次只开一条性能假设；先 CodeArts `-e` 再交平台采时  
- 未测前不要叠双缓冲 + 新 dtype 大改  

## 协作围栏

1. **不要改** `people/hongwei-2026/`；各写 `people/<自己>/`  
2. 共享台账只动本题 `tasks/active/PReluGradUpdate_2329/`  
3. 改默认试交包前先在 STATUS / 本页打招呼  
4. 云上每次构建成败都打 `EXP_*_logs.tar.gz` 下载归档  

## 下一闸

- hongwei：平台交 P100 采时；若仍 ms 级 → 查 aclnn 重复分配 / 开 P101（有界）  
- 队友：可读 EXP019；有独立性能假设先写 path-ledger 再动手  
