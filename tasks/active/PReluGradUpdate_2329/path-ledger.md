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
| P015 | 2026-11-02 | team | 功能 | 修 Compare/Select mask 通路 | 恢复 Select(g,wg)/(0,xg)；CompareScalar+PipeBarrier；mask 按 256-bit 对齐 | 本迭代 | `EXP_P015_…162823` 仍 WA **97/192**≡恒负支；Compare 仍失效 | 进行中(未过) | — |
| P016 | 2026-11-02 | team | 实现 | Codex 收紧 UB/单缓冲 | `BUFFER_NUM=1`；coef 21/29；usable 80%；同步 design_final | 本迭代 | `EXP_P016_…170730` 仍 WA **97/192**；与 P015 同（不修 mask） | 采纳(UB) / 未过样例 | — |
| P017 | 2026-11-02 | team | 功能 | Select dst/src1 别名踩负支 | 合入 Claude 重写：独立 neg/fg buffer + cmpCount 64 对齐 + DataCopyPad | 本迭代 | `EXP_P017`：**SyncFunc 未声明**，未编过 | 进行中 | — |
| P018 | 2026-11-02 | team | 环境 | 换掉 SyncFunc | `SetFlag`/`WaitFlag` S→V；记 EXP006 | 本迭代 | 已改 workdir；待云端重编 `-e` | 进行中 | — |
| P019 | 2026-11-02 | team | 环境 | P018 实际编译了旧 code 副本 | 加 `verify_codearts_source.sh`，构建前比对 workdir/build kernel SHA256，禁止残留 SyncFunc | 本迭代 | EXP_P019 verify **PASS**；编挂在 bf16 Cast | **采纳(门禁)** | — |
| P020 | 2026-11-02 | team | 环境 | bf16 禁止 Cast（EXP002 再现） | bf16 位扩展/RNE；half 仍 Cast；记 EXP007 | 本迭代 | 已改 kernel；待云端 `-e` | 进行中 | — |
| P021 | 2026-11-02 | team | 交付 | 先交 tip 工程试平台 | 打顶层 `code/` ZIP（beaa00e）；排除 pyc/build | 0.5h | `deliverables/PReluGradUpdate_2329_code_best.zip` SHA256 `A7C800F6…7F973D` | 采纳(试交) | — |
| P022 | 2026-11-02 | yuhongming-2026 | 功能 | CompareScalar 致 mask 全 0 | PR#1：`Compare`+零；八参 Select+BinaryRepeatParams | 本迭代 | 已合 main（座位代码）；八参 Select 未云测，试交以 P025 为准 | 部分采纳 | EXP008 |
| P023 | 2026-11-03 | hongwei-2026 | 功能 | 吸 P022 比较、保可编译 Select | hongwei：`Compare(+zero)` + **count** Select；禁八参 | 本迭代 | EXP_P023 仍 `bf16 type cast` | 进行中 | EXP009 |
| P024 | 2026-11-03 | hongwei-2026 | 环境 | if constexpr 藏 Cast 不够 | `DtypeBridge` 全特化；默认 `PRELU_ENABLE_BF16=0` 只编 fp16/fp32 | 本迭代 | EXP_P024 仍 bf16 Cast（hash=旧 fp16/bf16） | 进行中 | EXP010 |
| P025 | 2026-11-03 | hongwei-2026 | 环境 | half Cast/bf16 符号仍进后端 | 默认 **仅 fp32**：`PRELU_ENABLE_FP16/BF16=0`；Cast/bf16 全 `#if` | 本迭代 | Build OK；样例 dx/update **0/192 PASS** | **采纳** | EXP012 |
| P026 | 2026-11-03 | hongwei-2026 | 功能 | 平台4点 507035=缺低精度/核异常 | 开三 dtype；half/bf16 **无 Cast**；每 tile 重建 weight；WS 16MB | 本迭代 | EXP_P026 仍 bf16 Cast 三 key 全挂 | 部分废弃(bf16) | EXP014 |
| P027 | 2026-11-03 | hongwei-2026 | 环境 | bf16 进编译单元必炸 | **只开 fp32+fp16**；`PRELU_ENABLE_BF16=0`；half 标量无 Cast | 本迭代 | EXP_P027：fp16+fp32 仍双挂 Cast | **废弃(混编)** | EXP015 |
| P028 | 2026-11-03 | hongwei-2026 | 环境 | half 路径污染同 TU 的 fp32 | 默认回到 **仅 fp32**（对齐 P025）；修 verify 误报 | 本迭代 | 已改；以 P025 为可交基线 | **采纳** | EXP015 |
| P029 | 2026-11-03 | hongwei-2026 | 功能 | 显式特化/ToFloat 污染各 key | 删 DtypeBridge；float/half **原生**；bf16 仅 if constexpr 位转换；开三 dtype | 本迭代 | EXP_P029 云测 PASS；平台 case4 仍 507035 | 部分采纳 | EXP016 |
| P030 | 2026-11-03 | hongwei-2026 | 功能 | case4=低精度运行时炸 | half 改 Cast↔fp32；bf16 补 S_V/S_MTE3；禁半精度成员污染 | 本迭代 | EXP_P030 云测 PASS；平台 case4 仍 507035 | 部分采纳 | EXP017 |
| P031 | 2026-11-03 | hongwei-2026 | 功能 | case4=per-channel weight 的 `Duplicate` 从非 32B 对齐 UB 偏移启动 | 每段未对齐前缀用 `SetValue`；仅从 8-float 对齐下标 `Duplicate`；补 innerSize=1/3/5/7/35/8255 回归 | 本迭代 | 平台 **1–5 Pass**（原 case4 已过） | **采纳** | EXP018 |
| P032 | 2026-11-03 | yuhongming-2026 | 功能 | 座位代码仍含未对齐 `Duplicate(weightLocal[localOffset])`，会复现 case4 507035 | 吸 P031/EXP018 进 `people/yuhongming-2026/.../BuildWeightVector`；补对齐静态回归 | 本迭代 | 本地语义 PASS；待 CodeArts | 进行中 | EXP018 |
| P100 | 2026-11-03 | hongwei-2026 | 性能 | ms 级固定开销来自未使用 16MB workspace；单核内 per-channel 权重展开又串行放大 | workspace=0；全核 tiling；channel 内 `Muls`；**补丁** `progress+=currentNum` + host `tileLen<=inner` | 0.5d | EXP_P100 云测三 key + 样例 **0/192 PASS**；待平台采时 | 进行中 | EXP019 |

## 性能道路预声明（未到阶段勿开）

| ID | 触发条件 | 默认策略 |
|----|----------|----------|
| P1xx | 功能精度基线采纳后，全核 case < 95% TBE | 标准 tiling/全核等有界优化；触止损写 DEC |
| P1yy | 小 shape <10us 且差 ≤3us | **仿真图 + 分析**，默认不抠微秒 |
