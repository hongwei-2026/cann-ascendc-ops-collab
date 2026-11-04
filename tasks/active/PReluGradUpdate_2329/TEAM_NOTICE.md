# 当前候选 — P105（2026-11-04）

当前 hongwei 席位实时路径已切换到 **P105 待测候选**：只把原权重索引、GM 读取与标量解码块移到 Compute 入口、输入 DeQue 之前，尝试与已发出的输入搬运重叠。Host、UB、分块、数值计算和同步保持 P103。源码精确搬移及原两组本地检查通过；CANN 编译、设备精度与平台收益尚未验证。

P103 `47db93e` 是已测参考（历史 42.89），P104 继续不采纳。用户说明无法提供 profiler，要求继续小步试验；不再把 profile 作为前置阻塞。下一闸：固定源码/构建指纹 → 三 key 编译与精度 → 平台计时 → 决定保留或回退。仅本地修改，不提交、不推送。入口：`deliverables/SUBMIT_LINKS_P105.md`；实验：EXP025。

以下为 P104 回退时的历史记录；其中“默认 P103”和“先获取 profile”以本节后续状态为准。

---

# 队友同步 — P104 平台变差已拒（2026-11-04）

本次执行补充：实际 workdir 的 Kernel、Host tiling 和两份对应检查脚本已恢复 P103 `47db93e`，四文件内容和本地检查已复核。回退后的平台耗时尚未复测；历史 42.89 不代表当前成绩保证。TBest 也发生变化，分差不能当作纯代码退化幅度。下一步先做绑定源码/产物的同环境对照和设备分析，不再按少屏障推断加速。本次只改本地，不提交、不推送。

@`yuhongming-2026` @`JadeYara` — 请先读本页再改 Kernel / 交平台。

## 一句话

**默认仍是 P103**（tip `47db93e`，平台 **42.89**）。  
P104 云测功能过，但平台 19:59 前三点 **8.15/8.22/8.25μs** 劣于 P103 的 **7.36/7.30/7.24μs** → **DEC002 不采纳**。  
**问题：性能仍不够**（约 1.5～2× TBest）；微调度道路未突破。

## 现状

| 项 | 状态 |
|----|------|
| 正确性 | 平台 **1–5 Pass**；云测 0/192 |
| 默认 | **P103 / 42.89**（7.36/7.30/7.24/5.83/6.07μs） |
| P104 | 提交 1132450 Pass；8.15/8.22/8.25/5.79/5.99μs → **废弃** |
| 阶段 | **性能优化中 · 缺口仍大** |

## 已知问题（请对齐认知）

1. **公开小 shape 仍慢**：差 >3μs，按 `02-perf-gate` 属 B 类边缘，禁止无 profile 死磕。  
2. **已试道路不够**：P100（workspace=0 + 全核 + channel `Muls`）把 ms 打到 μs；P102（对齐 DataCopy）平台变慢已 DEC；P103（零缓冲复用）分数略升但相对 TBest 无质变。  
3. **缺证据**：多数平台图无源码 hash；无稳定 A/B、无流水线 profile，不能凭感觉叠双缓冲。  
4. **隐藏点 / 大 shape**：是否 A 类门禁未用公开五点代替；需要任务书 case 与同机 TBE 表。

## 正确性基线（冻结）

1. weight `Duplicate` 32B 对齐（P031）  
2. Select 用 **count**；禁八参 `BinaryRepeatParams`  
3. bf16 禁 `Cast`；half 仅 `if constexpr`  
4. `progress += currentNum`（channel tile）

## 默认包 P103（在 P100 之上）

- P100：`workspace=0`；全核；channel 内 `Muls`；`progress += currentNum`  
- P103：update Select 前移，复用 dx 零缓冲，少 Duplicate/屏障（EXP023）  
- **不要**交 P102 / 外部模板 / 八参 Select 包

## 协作围栏

1. 勿改 `people/hongwei-2026/`；各写自己座位  
2. 新性能假设先写 `path-ledger` 一行，再动手；一次一条  
3. 云测日志进 `/mnt/workspace/exp_logs/prelu2329/`，下载归档再认结果  
4. 触止损写 `decisions/DEC*.md`，禁止假 PASS

## 下一闸

- 先分清 A/B 类 case；A 类冲 TBE 95%，B 类优先仿真图+分析  
- 有新假设：ledger → CodeArts `-e` → 同机平台采时（带指纹）  
- 无 ≥2% 同机增益或超时 → DEC，回退默认 P103  

依据：STATUS · EXP019 · EXP022/DEC001 · EXP023 · 截图 `explorations/screenshots/EXP023-platform-20261103-164642.png`
