# PReluGradUpdate_2329 下一阶段建议（DeepSeek 调研）

日期：2026-11-08  
输入材料：`03-当前进度与DeepSeek调研交接-PReluGradUpdate_2329-20261108.md`、`J071-standard-code.zip`（52 文件，SHA256 `34e7099f…`）、J069/J070/J071/J072 本地证据、J067 真 TBE 参考证据、平台五点记录。  
本文只写建议，不改源码、不改生产、不跑设备实验、不提交平台。文中的所有"官方"结论都以交接文档中的用户截图/用户口径为限，未取得平台提交快照独立复核。

---

## 0. 一页结论（先看这里）

1. **J072 不晋升是正确的；但在查清 J072 的 BF16 旧路径回退之前，不应继续扩展 General 路径。** 静态核对已把 J072 的改动缩到两类：General 内核/UB 追加、Host 的 General UB 常量一行；旧路径的 Host 选路与 tile 逻辑未改。与此同时，J072 中 BF16 旧路径函数体是**逐字节相同**的，变慢的 15 个输入又全是旧路径 BF16，且 A/A 控制 90/90 稳定。这使"旧路径代码变慢"基本被排除，但**包级布局/加载上下文 vs 运行态/测量偏差**仍未分开，不能预设包布局是根因（J064 已给过"ELF 内地址相同 ≠ 运行时加载地址相同"的教训）。
2. **当前最高性价比的结构方向不是再抠 Compare/Select 算术，而是"小 shape 的固定开销 + 通用广播的源展开方式"。** General 路径在 3.4K–7.7K 元素的小用例上会启动 32–39 个核（Host 按 total/40 均分后再对齐），而公开 AscendC 经验明确给出每核约 60–100 ns 的 launch ramp，小/中任务满核白付约 1–3 µs；这正好是 General 小用例相对 TBE 的主要缺口量级。先把 Host 核数分档作为单变量实验，风险最低、无需改数学。
3. **General 的通用索引+Gather 应降级为最后兜底。** 公开的 ops-nn/opbase 广播实现把规律广播放在 CopyIn/Brc 阶段（NDDMA、one-dim、UB broadcast、noncontiguous 四条路径），不是对每个 lane 建索引再 Gather；这与交接文档要求的"identity / 标量 / 1D 通道 / 少广播轴 / 其余兜底"最小分派完全一致。建议按结构类别逐类加快速路径，一次只验证一类，不做 shape 硬编码。
4. **低精度优化必须服从真 TBE 数值链，不能"顺手"简化。** `dx` 量化回读、`dx*0` 进 update、Inf/NaN 传播都是契约的一部分；J071 的 FP32 双槽流水里那段 `if constexpr(half)` 原生半精度分支**不是**真 TBE 链（仅为未被实例化的遗留分支），不能拿来扩展 FP16/BF16 流水。
5. **官方成绩还没有可用于归因的新证据。** J069 官方五点 `7.55/7.43/7.56/5.77/5.83`，相对 J035 `7.15/7.29/7.11/5.88/6.06` 是"前三点慢、后两点快"，不能说整体突破；J070/J071 均无已核验官方成绩；J072 不提交。下一步官方动作只能是一次单变量试交 + 用户截图，不能把本地输入绑定到点号，也不能按耗时猜隐藏 shape。

---

## 1. 题目与合同再核对（事实）

### 1.1 题目要求

用户给出的题面：参考内置 TBE `PReluGradUpdate`，Ascend C 实现功能一致算子；支持原算子对应的所有数据类型、数据格式；比较方式从二进制比较改为逻辑值比较；必须泛化；精度按 CANN Judge 默认阈值（float 1e-4，half 1e-3，int 0）；隐藏用例用泛化数据做功能/精度/性能验证。  
本地 `E:\suanzi\cann-ascendc-ops-collab\docs\collaboration\02-perf-gate.md:20-28` 记录了本题性能条款：A 类全核场景要求 ≥ 原 TBE 95%；小 shape（10 µs 以下相差 3 µs）可用仿真图与分析结论走 B 类，不死磕微秒。

### 1.2 真 TBE 支持面（J067 参考证据）

`J067/reference-evidence/prelu_grad_update.py`（本地副本，行号见 `J070/reference-lines.json`）：

- dtype：`valid_dtypes = ("float16", "float32", "bfloat16")`；**没有 int**。题面的 int 0 是平台通用精度条款，不是本算子支持列表。
- format：`valid_formats = ("ND", "NCHW", "NC1HWC0")`；NC1HWC0 的实际格式值可带子格式位（J070 用 `GetPrimaryFormat` 处理）。
- 权重映射不是简单按通道：`broadcast_weight_shape` 按 TBE 分支顺序处理 `len(w)==len(x)-1`、x 为 4D、NC1HWC0 下 5D/1D 等；J071 的 `shape_utils.h` 已按同一分支顺序实现（见 `J071 code/op_host/p_relu_grad_update_shape_utils.h`）。
- rank 上限 8（`prelu_grad_update_tiling.h` 的 `PRELU_GENERAL_RANK=8` 注释引用 CANN `para_check.py:43,651-653`）。
- 真数值链（`compare_zero_and_select` + `prelu_grad_upate_compute`）：
  - fp16/bf16 先 `cast_to float32`，权重/特征/梯度按 `shape_max` 广播；
  - `zero_one = x>0 ? 0 : 1`，`one_zero = x>0 ? 1 : 0`；
  - `res_dx = one_zero + zero_one*w`；`res_update = zero_one*x`；
  - `dx = res_dx*g`；`update = res_update*g`；
  - 低精度：`dx` 先按 dtype 量化（half 用 `cast_to`，bf16 用 `round`），再 cast 回 fp32，`dx*0` 加到 `update`，最后 `update` 量化回 dtype。
- 因此：`where` 简化、掩码乘法、原生 half 算术、丢掉 `dx*0`、改变运算顺序，都必须先给出数据域等价证明；旧 where 证明已作废。

**条件**：以上是 CANN 9.3 legacy TBE 源码事实；J071 的 OpDef 已注册 ND/NCHW/NC1HWC0 × fp16/fp32/bf16（`J071 code/op_host/p_relu_grad_update_def.cpp`），所以"格式支持"这块不要再当缺口；真正的缺口在 General 路径的**物理广播执行效率**，而不是"没注册格式"。

---

## 2. 当前版本链（事实/状态）

| 版本 | 结构 | 本地结论 | 官方 |
|---|---|---|---|
| J069 | 连续 Gather 快路径（bounded C/medium S），保留真 TBE 数学 | 53 个 Gather 输入 318/318 轮改善，输入级中位提速 86.1%；74 个未改输入 444/444 控制稳定 | 五点 Pass，0.00%；`7.55/7.43/7.56/5.77/5.83`（用户截图，提交 ID 未知） |
| J070 | General compact-window 广播兜底，修 NCHW/NC1/Inf 等 | 63 个格式/广播/空维/特殊值提议中内置 TBE 接受 61 个，标准/B 逻辑差异 0；General 12 例对 TBE：仅互补广播 FP16/BF16 快约 6.3%/7.7%，其余慢 22.5%–170.1% | 未提交/无成绩 |
| J071 | J069+J070 + FP32 split-C tileLen=4096 输入输出双槽 | 67 个公开 ACL 提议中 TBE 接受 65 个，精度 1067 组全过；11 个启用输入 66/66 轮快；目标 `[127,8192,1]` 13.6654→11.8605 µs（+13.21%）；762/762 严格控制 | 未提交/无成绩 |
| J072 | J071 + General 三路输入集中搬运（独立 compact/raw，先算窗口、一次等三路 DMA） | General 仅窗口边界 3 例 +6.6~7.0%，其余多为 −2.2%~+1.5%；旧路径 BF16 15 输入 90 轮回退，最差 `[127,64,1]` −21.55%；A/A 控制 90/90 稳定 | 不晋升/不提交 |

官方对照：J035 `7.15/7.29/7.11/5.88/6.06`，J056 `7.21/7.16/7.19/5.58/5.84`。J069 相对 J035 前三点慢 0.40/0.14/0.45 µs、后两点快 0.11/0.23 µs；只能写"有得有失"，不能写"五点提升"。

---

## 3. J072 失败：已读事实与可区分假设

### 3.1 已读事实

1. J072 相对 J071 的源码差异见 `J072/general.diff`：
   - 每个 General 输入独立 compact/raw 缓冲（`compact1_/compact2_/raw1_/raw2_`），集中 3 路 DMA、一次 MTE2 等待，再做原 Cast/Gather/Compute；
   - 原 General 的 window 函数、索引公式、Compute 真 TBE 链逐字保持；
   - 新增缓冲追加在原分配之后，原二元输入 256B 相位不变。
2. Host 差异只有一行：`J072/candidate/code/op_host/p_relu_grad_update_tiling.cpp`（对比 J071）把 General UB 估算从 `37*tile+512(+6*tile)` 改为 `45*tile+512(+10*tile)`；**旧路径选路/tiling 代码未改**（`PReluGradUpdateLegacyTilingFunc` 在 General 分支之前返回）。
3. 机器码（`J072/evidence/function-bodies.json`）：
   - `B-vs-A bf16`：60/63 相等，只有编码 key20 不等；key20 对应 BF16 的 General 编码 key（J069 DEC 已说明 UI_LIST 编码是列表序号，General BF16 为最后一个序号）。
   - `B-vs-A fp32`：60/63 相等，key18 不等（FP32 General）。
   - `B-vs-A fp16`：45/63 相等，key1/4/7/10/13/19 不等（6 个，含 FP16 General key19）。
   - `C-vs-A` 与 `standard-vs-B` 全等。
4. 旧路径回退（`J072/result-summary.json` 的 `legacy`）：
   - 15 个输入、90 轮，全部 bfloat16；A/A 控制 90/90 通过；
   - 最差 `[127,64,1]` bf16 wlen=64：J071 5.972625 → J072 7.265250 µs（−21.55%）；其余最差 `[83,64,1]` −13.99%、`[4097,128,1]` −6.47%、`[8041,64,1]` −6.24%、`[2680,64,1]` −4.73%、`[2681,64,1]` −4.67%、`[17,4097,1]` −4.56%、`[4097,64,1]`（wlen=1）−3.89%、`[4097,64,1]`（wlen=64）−3.83%、`[2679,64,1]` −3.72%。
5. General 结果（同上 README）：J072 相对 J071 只在窗口边界 3 例稳定 +6.6~7.0%，互补广播/NC1 等多数为负；对 TBE 只有互补广播 FP16/BF16 更快。

### 3.2 由事实收缩出的假设（不能预设根因）

- **H1 包级布局/代码生成**：J072 的 B 包比 A/C 多出 General 函数与 UB 分配；BF16 旧路径函数字节相同，但整个 `.o`/包的 section 大小、函数地址、重定位、加载布局变了。J064 的教训是"ELF 内地址相同不等于运行时加载地址相同"，所以这里不能写"加载/放置是根因"，只能把它作为一个待排除类。  
  旁证（假设，不证明因果）：J069 只加 Gather 快路径，官方五点也出现"前三点慢 0.40/0.14/0.45 µs、后两点快 0.11/0.23 µs"；J072 只加 General 代码，本地旧路径 BF16 出现 15 输入回退。两者形态一致地提示"新增路径的包/布局可能影响未改路径"，但都缺少同进程、同输入的包级对照，不能当结论。  
  证据方向：J071/J072 BF16 `.o` 的旧路径函数地址/段大小对比；同包 padding 对照；分离 General 编译单元的构建。
- **H2 运行态/测量偏差**：A/A 控制通过说明不是基线漂移；但 J072 的 B 内核在 General 用例中会走更重的 DMA/UB 路径，旧路径窗口是独立循环还是同一进程前后交错，需要按 campaign 脚本核实。若旧路径回退只在 General 运行之后出现，则属于运行态（cache/功耗/队列），不是包布局。  
  证据方向：同一 J072 构建、同一旧路径输入，在"先跑 General"与"不跑 General"两种顺序下对比；新进程重复。
- **H3 Host/UB 残留**：静态 diff 已把旧路径 Host 排除（只有 General UB 常量一行），但正式实验前仍应复核 15 个失败输入在 J071/J072 下的 TilingData/blockDim/tileLen/UB 估算是否逐字相同；若不成立，说明还有未读到的 Host 分支。
- **H4 纯测量/环境**：预注册的新进程 A/B 重复，不筛选、不改门槛、不选择性重测。

### 3.3 最小归因实验（P1，优先于任何 General 优化）

按"先静态、后单变量、最后才判类"的顺序：

**P1-a（只读，零设备）**
- 复核 15 个失败输入的 Host tiling：J071 vs J072 的 TilingData、blockDim、tileLen、UB 估算逐字段对比（用本地 UT/CPU Host 路径即可）；
- 解析 J071/J072 BF16 `.o`：旧路径函数地址、大小、段大小、重定位；确认 key20 是 BF16 General、旧 key 未变；
- 若旧路径函数地址/偏移完全一致而包大小不同 → H1 优先；若 Host 字段有差异 → H3；若都无差异 → H2/H4。
- 输出一份"事实表"（输入/函数/key/地址/Host 字段），不写根因结论。

**P1-b（单变量，优先）**
- 控制 A：取 J071，在 BF16 翻译单元中加入与 J072 General 代码**同量级但无 General 路由**的 dummy 代码/缓冲，构建后跑旧路径 15 输入 + 127 输入控制；
- 判据：若最差输入回退 ≥5%，且旧路径函数/路由未变 → 支持 H1（包级/代码生成敏感）；若干净 → H1 削弱，转 P1-c。
- 注意：这是"加 dummy"不是"复制 General 逻辑"，只测代码/包大小与布局；不改变旧路径 Host/数学。

**P1-c（单变量）**
- 用 J072 构建，但让 Host 对 General 输入直接拒绝（General 不执行），只跑旧路径 BF16/FP16/FP32 15 输入 + 127 控制；若回退仍在 → H1；若消失 → H2（General 执行影响旧路径窗口）或 H3。
- 若旧路径函数字节/地址在 P1-a 中完全一致，P1-c 是区分 H1/H2 的关键。

**P1-d（预注册对照，可选）**
- 同一 J071/J072 构建，新进程再跑一次旧路径 BF16 15 输入（A/A 先验证）；若不复现 → H4/运行态；若复现 → 系统性。只做一次，所有结果保留，不重测刷门槛。

**停止条件**
- P1-b 出现旧路径最差输入 ≥5% 回退且旧路径 Host/函数未变 → 支持 H1，**停止继续找"某条指令变慢"**，J072 结案不晋升；
- P1-b 干净、P1-c（General 不执行）仍有 ≥5% 回退 → 也是 H1（与 General 是否执行无关）；P1-c 回退消失 → 转 H2，检查 General 执行/测量顺序，不再追 H1；
- 若 P1-a 发现 Host 字段差异 → 先解释 Host，再谈性能；
- 若新进程和所有对照均不复现 → 记 H4，保留 J071，后续所有 campaign 强制 A/A 控制，不再追 J072；
- 任一实验不得改控制门槛、不得选择性重测；全部 15 输入的结果都要保留。

---

## 4. 性能差距的结构分析（事实 + 公开依据）

### 4.1 General 路径的真实成本构成

`J071 code/op_kernel/p_relu_grad_update_general.h`：

- tile 固定 2048 lanes（`general_tiling.h`），每个输入每 tile：
  - `Expand` 对非 identity 源做：`Muls`（lane×last stride）、`Adds`（base−lo）、对每个 `rank−1` 轴做 `Adds→Muls→ShiftRight→Muls→Add`（系数为 0 的轴跳过），再 `Gather`（约 5 条/轴 + 2 条 + Gather）；
  - 3 个源各做一次，再进 `Compute`（真 TBE 链约 12–15 条向量操作 + 多次 `PipeBarrier`，低精度再加 4 次 Cast）。
- 每个 General 输入都要独立 compact copy；J072 只是把 3 路 copy 批量化，索引/窗口/Gather 结构未变，所以窗口边界仅 +6.6–7.0%，其余无收益，这与代码结构一致。
- Host（`op_host/p_relu_grad_update_tiling.cpp:375-379`）：
  - `perCore = ceil(total/40)`；`blockSpan = align(perCore, 32/width)`；`used = ceil(total/blockSpan)`。
  - 实测估算：total=255 → 32 核；3360/4641/7735（FP32）→ 39 核；572339 → 40 核。
- 公开经验（`cannbot-skills` AscendC shuffle/index 指南）：每核有约 60–100 ns 的 block 分发/ramp；小/中任务满核白付约 1–3 µs；建议按规模分档（极小按行数 1 行/核、中 [4,16]、大满核），且档位要在目标硬件实测。  
  → **条件**：这是公开文章的经验值，不是 910C/本题实测；只能作为"假设 + 需本地验证"的依据。

### 4.2 公开实现给出的方向（不是可直接复制的代码）

- ops-nn 上游 `activation/p_relu_grad_update` 的 arch35 实现（新版架构，非 910B legacy）：
  - kernel 入口 `p_relu_grad_update_apt.cpp` 用 `BroadcastSch<schMode, OpDag>` + `CopyInBrc`，计算走 `PReluGradUpdateDAG` 的微 API；
  - Host `prelu_grad_update_tiling_arch35.cpp` 用 `BroadcastBaseTiling` 生成 schMode；
  - 说明官方公共实现把"广播"作为 copy-in 调度的一部分，而不是对每个 lane 建索引再 Gather。
  - 来源：[ops-nn arch35 kernel 目录](https://gitcode.com/cann/ops-nn/tree/774c94bcb7a4836e8f5a1a18e6e2b75da481394c/activation/p_relu_grad_update/op_kernel/arch35)、[arch35 tiling](https://gitcode.com/cann/ops-nn/blob/774c94bcb7a4836e8f5a1a18e6e2b75da481394c/activation/p_relu_grad_update/op_host/arch35/prelu_grad_update_tiling_arch35.cpp)。**适用条件**：arch35 不是 910B 可直接使用的 API，只借其调度结构，不复制代码。
- CANN opbase 广播调度器（910 系列公共框架）已把路径分成多条：`broadcast_sch_nddma.h`（NDDMA CopyInBrc，内联轴 ≤4）、`broadcast_sch_one_dim.h`、`broadcast_sch_ub_broadcast.h`、`broadcast_tiling_noncontiguous.h`（非连续兜底）。  
  来源：[opbase atvoss/broadcast 目录](https://gitcode.com/cann/opbase/tree/master/pkg_inc/op_common/atvoss/broadcast)、[broadcast_sch_nddma.h](https://gitcode.com/cann/opbase/blob/master/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_nddma.h)。  
  **对我们的含义**：General 路径的"所有非 identity 源都走索引+Gather"是兜底，不是唯一解；至少应补"线性窗口直接 DMA"和"1D 通道权重 scalar/Duplicate"两类。
- 索引类算子公开指南（Ascend agent-skills）：Gather 源数据最大 32 KB、索引是 uint32 字节偏移，按行结构选 GatherPath/UBPath/ElementPath；广播加载只搬去重后的源行、规律重排优先硬件 pattern、repeat-stride 直接消费。  
  来源：[index-tiling.md](https://raw.gitcode.com/Ascend/agent-skills/raw/582e1ee95f89c1c53c0b2004624024eb6761f55a/skills/ascendc-operator-design/references/index-tiling.md)、[AscendC 重排/搬运指南](https://gitcode.com/cann/cannbot-skills/blob/eded7f04713d8730a70267645a8b745aadca7253/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_shuffle_patterns.md)。  
  **注意**：以上是公开社区/官方技能文档，按"技术参考"使用，不是本题合同；其结论需在 910C/本题复测。

### 4.3 官方五点与本地收益脱节（事实边界）

- 平台可见五点 TBE 基线 11/03 为 `4.13/4.16/4.23/3.99/4.01 µs`，11/04 为 `3.84/3.75/3.83/3.93/4.01 µs`（日期不同，不能横向拼）；J035/J069 五点均在 5.7–7.6 µs，属"小 shape/B 类边缘"区间。
- J069 本地 Gather 大胜（53 个形状中位 86%）对应的官方五点只出现"前三点略慢、后两点略快"，说明**本地 shape 与官方点不是一一对应**；J070/J071 的 General/流水收益更无法自动映射官方点。
- 结论：在官方 point shape/dtype 未知前，所有"提升官方分"的说法都只能写成待验证；不能把本地 `[127,8192,1]`、`[8,64,3]` 等绑定到点号。

---

## 5. 建议方案（按优先级，2–3 条）

### P1（最高优先）：把 J072 回退归因做完，再决定 General 去留

见 §3.3。这里只补"为什么必须先做"：

- 若 J072 回退是包级布局类，则说明当前 General 方案的"代码体积/UB 分配"会连带影响不相关的旧路径；未来任何 General 扩展都要先做包级 A/B，否则会把旧路径回退误判为算法问题。
- 若是运行态类，则 J072 的 General 收益（窗口边界 6–7%）不可靠，继续优化 General 会重复踩坑。
- 若所有对照都不复现，则 J072 属测量偏差；保留 J071，后续 campaign 必须 A/A 控制，不再投入 J072。
- 成功判据：给出"旧路径代码/函数地址/Host tiling 是否一致"的事实表，并至少一个单变量实验能复现或排除 ≥5% 的回退；否则记为未决，不写根因。

### P2（性能主方向）：General 最小分派 + 小 shape 核数分档

目标：把 General 从"所有非 identity 源都建索引+Gather"改成"按合法结构分派"，同时先解决小 shape 的核启动摊销。按以下顺序，**一次一个变量**：

**P2-a Host 核数分档（不改 kernel/数学，先做）**

- 修改：仅 General 分支核数/`blockSpan` 计算。固定一条规则，例如先取 `tier = min(hardwareCore, max(4, ceil(total/2048)))`，再按 `perCore = ceil(total/tier)`、`blockSpan = align(perCore, 32/width)`、`used = ceil(total/blockSpan)` 重算；不按 shape 查表、不扫多组阈值。total 很小（≤约 8K）时把 32–39 核降到 4–8 核；窗口边界等大 total 不变（tier=hardwareCore）。不要只把 `used` 截小而不重算 `blockSpan`，否则会漏覆盖。
- 依据：公开 ramp 1–3 µs；当前 General 小用例的缺口（NC1 约 6–7 µs、NCHW 约 4–5 µs）与 ramp 量级吻合；J025/J026 的"减核不普遍有效"针对旧路径大 shape，不适用于"General 小 shape"这一限定条件。
- 精度/格式风险：无（只改数据划分，Host 分区仍连续、每核 span 重算；TilingData 字段不变）。
- 最小 A/B：General 12 输入（ND 互补/窗口边界/NCHW/NC1/空维等）× 6 轮；同时跑 127 旧路径控制。
- 通过/停止：≥6 个受影响小用例中位改善 ≥5%，且旧路径无 >2% 回退、精度全过 → 保留；否则回退，不扫其它阈值。窗口边界 3 例若变差 >2% 也回退。

**P2-b 结构性快速路径（按类别，逐个验证）**

先只加两类，均只对"来源的 tile 图像是线性/周期"成立时启用，Host 分类依据是 shape/stride/广播轴，不是具体 shape 名单：

1. **线性窗口直接 DMA**：若某源在 P* 对齐 tile 内的 compact 图像按 lane 顺序连续（identity 之外的块广播，如 g 沿 N 广播、w 沿 N 广播），跳过索引+Gather，直接 `DataCopy` 到目标 UB（按 base−lo 偏移）；保留 identity 现有路径。
2. **1D 通道权重 scalar/Duplicate**：g/x identity、weights 映射为 `[1,C,1,...]` 且 tile 可对齐到通道边界时，每个 tile 段只取一次标量权重，用 `Muls`/`Duplicate` 消费；inner 小、tile 跨多个通道时按段循环；不满足则回 Gather/J069 路径。
3. 其余任意广播仍走 General 现有索引+Gather 兜底（保持正确性基线）。

- 依据：opbase 的 one-dim/ub-broadcast/nddma 分路；index-tiling 的"规律模式优先 pattern、广播只搬去重行"；J069 的 Gather 路径已证明"权重展开/连续 Gather"对 bounded C/medium S 有效。
- 精度风险：无新增数学；仍需 actual TBE golden 全量回归（J071 的 67 个提议/65 个 TBE 接受 + 1067 组精度；若改动涉及低精度/广播，再补 J070 的 65536 特征位型 oracle），任何 logical mismatch 即回退。
- 最小 A/B：12 General 输入 + 12 个代表性的旧 Gather/old 输入；一次只开一类快速路径，比如先只开"线性窗口"，再单独开"1D 通道"。
- 通过/停止：受影响类中位 ≥5–10%，未受影响控制 ≤2% 回退；若单类收益 <5% 或实现复杂度导致 UB/代码体积显著增加（参考 J072 教训），停止该类，不硬堆。

**P2-c（暂缓）通用索引表/多轴优化**

- 不建议现在做"一次建全量 offset 表"或"按 rank 查表"：公开经验已表明每 launch 重建表/标量 SetValue 是负收益（64 元素 case 25 µs 中有效搬运 <1 µs；标量 4096 次 SetValue ≈17.7 µs）；General 现在已经是矢量生成索引，问题在分路和核数，不在"表建得不够快"。
- 条件：只有 P2-a/P2-b 后 General 仍慢 >2×，且 profile 证明索引指令占比 >30%，才考虑"per-period 预计算索引向量一次复用"。

### P3（低精度/流水扩展边界）：先算 UB 和依赖，再单候选

**已知事实（来自 J071 代码与 README）**

- FP32 双槽 4096 的 UB：`185,088 B` 使用 / `188,416 B` 可用 / 余 `3,328 B`。分解：grads/features/dx/update 四个双槽各 `2×16,384`，weights `16,384+256`，mask `4,352`，neg/fg 各 `16,384`，phase pad `256`。
- 低精度单槽（同 tile=4096）估算：5 个单槽 `5×8,192 = 40,960`，mask `4,352`，6 个 fp32 缓冲约 `99,328`，合计约 `144,640`；若把 5 个队列都改成双槽，加 `40,960` → 约 `185,600`，理论余约 `2,816 B`。  
  **这是估算，不是实测**；必须按 `InitBuffer` 实际尺寸逐项复核，且考虑 `ubSize` 的 8 KB 预留假设，不能直接写"可行"。
- J071 的 `if constexpr(is_same<T,half>)` 分支是**原生 half 运算**（`full_pipeline.h:226-291`），不是真 TBE 链；它当前没有被实例化（入口只建 `FullPipeline<float>`），绝不能直接用于 FP16/BF16 扩展。

**建议步骤**

1. **只读 UB/依赖账本**：列出 4096/3072/2048 下 FP32/FP16/BF16 的 `InitBuffer` 明细、双槽所需、尾块处理；核对 Host `split-C` 对低精度是否会产生 `tileLen=4096` 的可测形状（若没有，低精度流水就没有收益对象，直接停止）。
2. **单候选二选一**：
   - A（优先）：**FP16/BF16 split-C tileLen=4096 双槽**，但必须用真 TBE fp32 链替换 half 原生分支；UB 账本若复核后仍 ≤188,416 才做。
   - B（更稳）：**FP32 split-C 的 tileLen<4096 双槽**：现有 pipeline 的缓冲按 tileLen 线性缩放，UB 更宽裕；只放开 Host 的 `tileLen==4096` 限制，加尾块/边界检查。它不承诺官方收益，但风险低于低精度重写。
3. **低精度数值简化**：暂不做。`dx*0` 进 update 是 TBE 契约的一部分；任何"有限值才可省"的优化都要先有全 special-value/全位型等价证明和 actual TBE oracle，且净收益 ≥2%，否则不做。原生 half/where/掩码乘法都不允许。
4. 通过/停止：UB 预算超限、Host 无可测形状、精度/逻辑差异 0 中的一个不满足 → 停止；单候选若 <2% 或任何旧控制回退 >2% → 不晋升。

**流水边界结论**：FP32 4096 双槽已接近 UB 上限（余 3.3 KB），继续加缓冲不现实；非 4096 FP32 是低风险扩展；低精度 4096 双槽理论余量约 2.8 KB，但需要先证明 Host 有可测形状、且必须重写 half 分支为真 TBE fp32 链，优先级低于 P1/P2。

---

## 6. 官方验证与提交建议（跨方案，不单独占实验名额）

1. **J071 是当前唯一建议试交的独立标准候选**；J072 不混入、不提交。J071 的包 SHA、52 文件、与云端源码逐字节对应已在 README/evidence-meta 记录；平台上传仍由用户完成。
2. 提交前必须核对：包 SHA、入口与头文件 7 参/TPipe 一致（J065 旧入口 6 参/新头 7 参的教训）、编译目标 `ascend910b`、平台实际使用的入口不是缓存旧版本；只换头不换入口会再次编译失败。
3. 官方 A/B 必须单变量：J069（已有官方五点）→ J071（整包）；不要一次混入 J070/J071/J072 多变量后解释五点变化。五点截图/提交 ID 由用户保留；没有截图就写"无已核验官方成绩"。
4. 官方五点 shape/dtype 未知：不得按耗时反推，也不得把本地 45/127 输入映射到点 1–5；若要解释 J069 官方"前三点慢/后两点快"，只能说"与本地 Gather 快路径覆盖范围不一致"或"新增代码/包布局影响未改路径"这两个假设，并等待下一次单变量官方数据（J071 整包 vs J069 是下一个可对照点）。
5. 性能门禁上，官方可见点都在 5.7–7.6 µs、TBE 3.8–4.2 µs，属 B 类小 shape 边缘；按 `02-perf-gate.md` 的 A/B/C 分流，若后续没有 A 类 shape 证据，不应为纯小 shape 无限抠微秒，优先保证精度/泛化与可复现官方 A/B。

---

## 7. 不建议再做的事（避免重复已证伪/低收益路径）

- **不要**把 `where` 语义、掩码乘法、原生 half 算术、去掉 `dx*0` 当"简化"：真 TBE 链与旧 where 已在 J067/J068 纠正，J070 用 actual TBE oracle 验证过；任何改动都要先证明数据域等价。
- **不要**做微秒级"删屏障/合并 InitBuffer/只改 UB 相位"的微观优化：公开指南实测屏障删除 <±0.3 µs（噪声内），J060/J072 的队列/搬运合并也未形成整体收益。
- **不要**复制 J025/J026 的大 shape 20/32/40 核预算扫描：当时结论是"减少核数或 tile 不是普遍有效"；只有在 P2-a 限定的 General 小 shape 上做固定档位单变量。
- **不要**再试 DataCopy 对齐写回、持久零缓冲、TPipe 内部化等已被 P102/P104/J065 证伪或只在小范围有效的方向；J065 的价值是固定开销证据，不是新一轮 General 主线。
- **不要**把 J072 的窗口边界 +6.6–7.0% 当晋升依据：同轮旧路径 −21.55% 未解释，且 J072 General 其余 9 例仍慢于 TBE 25%–170%。
- **不要**在不知道官方点 shape/dtype 的情况下按耗时反推点号；不要以本地收益宣称官方提升；不要做未授权的平台登录/上传。

---

## 8. 证据与来源索引

### 本地材料

- 交接文档：`C:\Users\lin\.dsh\attachments\v1\files\2a\2af8a4907c81870f1b687c1d1f2885900f7d7d2e4c09ad2b3749234e50a66b3e\03-当前进度与DeepSeek调研交接-PReluGradUpdate_2329-20261108.md`
- J071 标准包：`E:\suanzi\J071-standard-code.zip`（SHA256 `34e7099fc0d84471b9fb78894cab2315216dc8bd077f07e80db46964e33ef85f`）；解包阅读目录 `E:\suanzi\PReluGradUpdate_2329-调研-20261107\J071-code\code\`
- J071 代码（相对 `code/`）：
  - `op_kernel/p_relu_grad_update.cpp:21-128`（entry/General/Gather/FP32 pipeline）
  - `op_kernel/p_relu_grad_update_general.h:10-124`（Init/Process/Window/Expand/Compute）
  - `op_kernel/p_relu_grad_update_general_tiling.h`（tile=2048、760B）
  - `op_kernel/p_relu_grad_update_gather.h:76-106`（权重展开+真 TBE 链）
  - `op_kernel/p_relu_grad_update_full_pipeline.h:140-169`（FP32 UB）、`226-291`（未实例化的原生 half 分支）、`292-438`（真 TBE fp32 链）、`453-471`（双槽 Process）
  - `op_host/p_relu_grad_update_tiling.cpp:237-239`（旧路径每 channel 核数）、`357-359`（旧路径返回）、`365-379`（General UB/核数）、`315-402`（General 规范化/路由）
  - `op_host/p_relu_grad_update_shape_utils.h`（TBE 权重映射分支）
  - `op_host/p_relu_grad_update_def.cpp`（3 dtype × ND/NCHW/NC1HWC0）
- 实验证据：
  - `...\experiments\J071\README.md`、`J071\result-summary.json`、`J071\DEC-J071.md`
  - `...\experiments\J072\README.md`、`J072\result-summary.json`、`J072\DEC-J072.md`、`J072\general.diff`、`J072\evidence\function-bodies.json`
  - `...\experiments\J069\DEC-J069.md`（Gather/官方五点/编码 key 说明）
  - `...\experiments\J070\visible-agent\RESULT.md`（General 对 TBE 表）、`J070\README.md`、`J070\reference-lines.json`
  - `...\experiments\J067\reference-evidence\prelu_grad_update.py`（真 TBE 链）
  - `...\experiments\J070-design\BROADCAST_WINDOW_PROOF.md`（窗口定理/单 hull 设计）
- 任务/门禁：`E:\suanzi\cann-ascendc-ops-collab\docs\collaboration\02-perf-gate.md:20-28`；模板 `materials\_unzipped\code\samples\reference.py`、`samples\case.json`。
- 平台五点截图（本地已读）：`tasks/active/PReluGradUpdate_2329/explorations/screenshots/EXP022-platform-20261103-154812.png`、`EXP023-platform-20261103-164642.png`、`EXP024-platform-20261104-195909.png`（后两者文件扩展名是 .png，实际字节是 JPEG；TBE 行分别见 11/03 与 11/04 两组基线）；J069 官方五点无本地截图文件，只有交接文档 §四的用户截图口径。

### 公开资料（外部，仅作技术参考；非本题合同）

- ops-nn 上游 arch35 广播调度实现：[kernel 目录](https://gitcode.com/cann/ops-nn/tree/774c94bcb7a4836e8f5a1a18e6e2b75da481394c/activation/p_relu_grad_update/op_kernel/arch35)、[arch35 tiling](https://gitcode.com/cann/ops-nn/blob/774c94bcb7a4836e8f5a1a18e6e2b75da481394c/activation/p_relu_grad_update/op_host/arch35/prelu_grad_update_tiling_arch35.cpp)、[apt 入口](https://gitcode.com/cann/ops-nn/blob/774c94bcb7a4836e8f5a1a18e6e2b75da481394c/activation/p_relu_grad_update/op_kernel/p_relu_grad_update_apt.cpp)
- CANN opbase 广播调度器：[atvoss/broadcast](https://gitcode.com/cann/opbase/tree/master/pkg_inc/op_common/atvoss/broadcast)、[broadcast_sch_nddma.h](https://gitcode.com/cann/opbase/blob/master/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_nddma.h)
- Ascend agent-skills 索引 tiling：[index-tiling.md](https://raw.gitcode.com/Ascend/agent-skills/raw/582e1ee95f89c1c53c0b2004624024eb6761f55a/skills/ascendc-operator-design/references/index-tiling.md)
- AscendC 重排/搬运类实现指南（固定开销、广播去重、核数分档、Gather 兜底）：[cannbot-skills shuffle_patterns.md](https://gitcode.com/cann/cannbot-skills/blob/eded7f04713d8730a70267645a8b745aadca7253/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_shuffle_patterns.md)

---

## 附录 A：J072 失败输入的快速核对清单（实验用）

对每个失败输入记录：shape、dtype、wlen、Host key/route、J071/J072 tiling 字段（totalNum/blockFactor/tileLen/rowsPerTile/splitBatchGroups/coreNum）、J071/J072 BF16 函数 key/地址/字节相等、A/A 控制值、6 轮 A1/A2/B/C 中位数、是否存在 General 先执行。  
已确认的最差输入：`[127,64,1] bf16 wlen=64`（J071 5.972625 → J072 7.265250，−21.55%）；完整 15 输入/90 轮见 `J072/result-summary.json legacy.exceptions`，不要只取最差值。

## 附录 B：General 核数示例（FP32，按当前 Host 公式）

| total | perCore | blockSpan(align8) | used (当前) | 建议初测 tier（示例） |
|---:|---:|---:|---:|---:|
| 255 | 7 | 8 | 32 | ≤4 |
| 3360 | 84 | 88 | 39 | ≤4–8 |
| 4641 | 117 | 120 | 39 | ≤4–8 |
| 7735 | 194 | 200 | 39 | ≤4–8 |
| 572339 | 14309 | 14312 | 40 | 40（不变） |

`used` 为按当前代码公式的估算；表格只用于说明量级，最终以 Host UT/设备实测 TilingData 为准。核数 tier 需在目标机上用一次预注册 ladder 选定，不能按 shape 列表硬编码。

## 附录 C：低精度流水 UB 账本（待实测复核）

| 项 | FP32 4096（已测） | 低精度 4096 单槽（估算） | 低精度 4096 双槽（估算） |
|---|---:|---:|---:|
| 5 个队列 | grads/features/dx/update 各 2×16,384；weights 16,384+256 | 5×8,192 = 40,960 | +5×8,192 = 81,920 |
| mask | 4,352 | 4,352 | 4,352 |
| fp32 工作缓冲 | neg/fg 各 16,384 | 6 个约 99,328 | 6 个约 99,328 |
| 合计 | **185,088** | **约 144,640** | **约 185,600** |
| 可用 | 188,416 | 188,416 | 188,416（需按实际 `InitBuffer` 复核） |

**警告**：低精度双槽约只剩 2.8 KB 余量，且 J071 的 half 分支不是真 TBE 链；没有逐项复核和 actual TBE 全量回归前，不得作为可晋升方案。 

---

（本文只做调研与建议，未修改任何源码、生产、台账或平台数据；未运行设备实验、未提交、未 push、未读取或发送凭据。）
