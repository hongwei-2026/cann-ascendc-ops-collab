# AI-3 优化分析 — PReluGradUpdate

> 视角：带宽瓶颈与 GM 流量、正确性边角、**先泛化测试再谈性能**、小 shape 何时停手写仿真。  
> 不写实现代码。硬件默认 910B（AIV），dtype：fp16 / fp32 / bf16。

---

## 1. 问题画像：这是带宽题，不是算力题

### 1.1 语义（验收口径，不是「像 PReLU 就行」）

对每个元素（`weights` 按通道维对齐，不是右对齐）：

| 条件 | `dx` | `update` |
|------|------|----------|
| `features > 0`（逻辑比较） | `grads` | `0` |
| 否则（含 `==0`、负值、NaN 比较为假） | `weights * grads` | `features * grads` |

- `weights`：`[C]` → reshape `[1, C, 1, ...]` 再广播到 `features`。
- `dx`、`update`：**与 `features` 同形**，**不做通道 reduce**。这是和部分 aclnn / 训练框架「dα 按通道求和」的最大语义差。
- 比较必须是**逻辑值比较**，禁止按 bit 模式当 mask。

### 1.2 算术强度估算（为什么先砍 GM）

一次有效计算大约：1 次比较 + 1～2 次乘 + 选择。  
GM 侧常态是 **3 读 2 写** 的大张量，外加可忽略的 `[C]`：

| 项 | 流量相对 `numel` |
|----|------------------|
| 读 `grads` / `features` | 2× |
| 读 `weights`（若按元素 gather 或先物化成同形） | 最多再 +1×，应压到 ≈0 |
| 写 `dx` / `update` | 2× |
| **下限（理想）** | **≈ 5× sizeof(T) × numel** |
| **踩坑上限** | 物化 broadcast 权重、写 mask、写中间结果、对 update 做 reduce 再广播 → 7～10×+ |

910B Vector 峰值远高于 HBM 能喂的速率。fp16/bf16 算术强度约 **0.3～0.5 FLOP/Byte**，fp32 更低。  
**结论：全核大 shape 的 95% TBE 门槛，几乎只取决于「有没有多碰 GM」和「双缓冲能不能盖住 5 路搬运」，不取决于换更花的指令序列。**

### 1.3 模板已经暴露的流量隐患

| 模板现状 | 对带宽的含义 |
|----------|----------------|
| Kernel / Tiling 空实现，`SetBlockDim(1)`，`totalNum=0` | 功能都没有，更谈不上全核带宽 |
| 只声明 `inputQueueX` + `outputQueueY`，`BUFFER_NUM=2` | 双缓冲**只是占位**；真做 3in+2out 时 UB 账必须重算，不能假装「已经双缓冲了」 |
| Tiling 扁平三字段：`totalNum / blockFactor / ubFactor` | 丢掉通道步长后，实现者很容易「整表 flatten + 每点 gather 权重」——正确但多一次不规则读 |
| `WS_SYS_SIZE=0` | **应保持**：本算子不该为 broadcast 开 GM workspace |
| `TYPE_SIZE=4`、`MIN_SPLIT_THRESHOLD=1024` 未接入 | 切核门槛写了但没用；切太碎会把 A 类打成假全核、B 类更超时 |

---

## 2. GM 流量最小化：只做「一次过、按通道复用」

### 2.1 必须守住的搬运契约

1. **单趟融合**：同一 UB tile 内读 `grads`+`features`（+当前通道的 `weights`），同时产出 `dx` 和 `update`，立刻写出。禁止「先算 dx 再扫一遍算 update」。
2. **权重不物化**：禁止把 `[C]` expand 成同形 GM 张量。`C` 通常远小于 UB，整段 `[C]` 常驻 UB 或按核切片驻留。
3. **按通道连续扫 inner**：布局为 `dim0=N`（可 >1）、`dim1=C`、`dim2+=inner`。对固定 `c`，`inner` 连续，该通道一个标量权重复用 `N×inner` 次。这是 910B 上最干净的 DataCopy 形态。
4. **禁止 GM 中间量**：mask、`zero_one`、`one_zero`、`result_dx` 都只活在 UB（TBE 伪代码是语义分解，不是要落盘）。
5. **update 不 reduce**：少一次原子/跨核归约，也少一次「[C] 输出」的错误形状。带宽和正确性同一条约束。
6. **fp16/bf16 在 UB 内升精度、不要先写成 fp32 GM**：golden 是「按 fp32 算再 cast 回声明 dtype」；升精度缓冲只在 UB。

### 2.2 UB 账（决定 tile，而不是先拍脑袋双缓冲）

910B UB 约 192～256KB。双缓冲时，每个元素至少要给：

- 输入：`grads`、`features`（+ 可选的权重复制行）
- 输出：`dx`、`update`
- 中间：比较 mask、cast 到 fp32 的工作缓冲（半精度路径）

粗算：fp32 裸 5 路 × 4B × 2 buffer ≈ 40 B/elem，再加中间很容易到 60～80 B/elem。  
半精度若「半精度驻留 + fp32 计算」，峰值占用更高。  
**ubFactor 必须从「UB − 队列预留 − 对齐 pad」反推**，模板里的 `ubFactor=0` 不能当默认。  
`BUFFER_NUM=2` 只在「单核 tile 数 ≥ 2 且 GM 占主导」时值钱；小 shape 开双缓冲只会吃掉 UB、减小 tile、增加 kick 次数。

### 2.3 切核怎么切才不增加流量

- **优先沿 `N×C×ceil(inner/align)` 切连续块**，每核一段连续 GM，避免交叉 stride。
- **不要按「元素 flatten 后 round-robin」切**：通道边界被切开后，每核都要重读/重索引 `weights`，尾块对齐也更脏。
- 全核门槛（A 类）：仅当 `numel` 大到填满 910B 全部 AIV、且每核数据量明显高于 kick 开销时，才 `SetBlockDim=coreNum`。模板常量 `MIN_SPLIT_THRESHOLD=1024` 是「别切太碎」的下限量级，不是性能目标。
- 尾核 `currentNum < ubFactor`：只多拷对齐 pad，不要为对齐再读一整份权重平面。

### 2.4 明确不值得做的「优化」

| 诱惑 | 为什么停 |
|------|----------|
| 先 GM broadcast `weights` 再当 elementwise | 多 1× 大读写，算术强度更差 |
| 为对标 TBE 逐步写 `zero_one`/`one_zero` 到 workspace | 教学拆步，带宽自杀 |
| 小 shape 上堆多流水、多 key、多分支 | kick + 编译路径变贵，B 类更差 |
| 追求「少写 update 零区」的稀疏写 | `update` 在正区必须是 0，仍要写满同形；稀疏写破坏连续 store |
| 为非连续输入写 stride kernel | `OpDef` 已 `AutoContiguous()`，设备侧按 ND 连续即可 |

---

## 3. 正确性边角：先这些，再谈 95%

### 3.1 逻辑比较，不是 bit 比较

- `features > 0`：`0`、`-0`、负数走「负支」：`dx = w*grads`，`update = features*grads`。  
  `features == 0` 时 `update = 0`，与正支写 0 **数值相同**，但 `dx` 不同（`w*grads` vs `grads`）。**零值是区分比较实现的探针，必须单独立 case。**
- NaN：IEEE 下 `NaN > 0` 为假，应走负支。不要用「看符号位」代替。
- +Inf 为正支；−Inf 为负支。
- 任务书把「二进制比较 → 逻辑比较」写成硬要求：实现和 golden 都必须是值比较。

### 3.2 形状与广播

| 边角 | 期望 |
|------|------|
| `C=1`，`weights=[1]` | 全体空间共用一个 α；最容易写成「忘了广播、当标量错位」或「按最后一维对齐」 |
| 近标量：`[1,1,1]`、`[1,C,1]`、`inner=1` 的 `[N,C]` | tiling 三字段会把「通道轴」吞掉；必须另存 `C`、`inner`、`outer`，或保证 flatten 后仍能还原 `c = (idx / inner) % C` |
| 高 rank：5D/6D，`[N,C,D,H,W]` | 广播形固定 `[1,C,1,1,1]`；`inner=D*H*W`。InferShape 不得按「输出 i = 输入 i」抄形状 |
| `grads` 与 `features` 不同形 | 应失败，而不是静默以 grads 形写 dx |
| `weights` 长度 ≠ `C`、或非 1D | 应失败 |
| rank&lt;2（reference 直接拒） | Host 校验 / InferShape 失败，不要在 kernel 里当 1D elementwise |

**InferShape 现状**：循环里 `output[i] ← input[i]`，`dx` 抄 `grads`、`update` 抄 `features`。二者同形时碰巧对；一旦泛化到非法或动态维，既不校验 `weights`，也不保证两输出都等于 `features`。这是功能门的 P0，不是性能项。

### 3.3 `C=1` / 近标量 / 高 rank 对 kernel 的具体坑

- **C=1**：`inner` 很大时看起来像纯 elementwise，但仍要读一次 weight；`inner=1` 时几乎是标量核，切多核只会全是尾块。
- **高 rank + 小 inner**：连续段短，DataCopy 粒度差，对齐 pad 占比高——这是精度/功能问题（尾块脏数据）先于性能问题。
- **大 C、inner=1**：权重几乎变成逐点表，和 features 一样长；此时仍不要先 expand 到 GM，按点取 UB 内 `weights[c]` 即可。

### 3.4 数值与 dtype

- 半精度：比较在值域（建议升 fp32 后比，或保证 `>0` 与 golden 一致），乘在 fp32，写回 round-to-nearest 与 Judge 默认阈值对齐。
- **bf16 被映射成 `half` tiling key（MODE_0）**：Host 把 `DT_FLOAT16` 与 `DT_BF16` 打到同一 key，Kernel 只实例化 `PReluGradUpdate<half>`。bf16 走 fp16 算子会在比较阈值、溢出、golden 位级上全面错。这是**正确性 P0**，不是「以后再加个 key 的性能优化」。
- 权重大小：α=0（负支 dx 全 0）、α=1（负支 dx=grads）、α&lt;0（合法 PReLU 变体）都要覆盖。

### 3.5 非连续：已经 AutoContiguous，测的是图，不是 kernel stride

- Device kernel **按连续 ND 写**即可，不要为 stride 开第二条计算路径。
- 泛化用例仍应构造「逻辑非连续」（转置、切片）的 Host 张量，确认图侧 contiguous 后结果与连续 golden 一致。失败点在 Host/图，不在「kernel 少写了 stride」。
- 不要把非连续当成性能专题。

---

## 4. 测试 / 泛化矩阵：任何 perf 开关之前必须绿

任务书：**必须泛化；验收用泛化数据**。性能专题启动条件（本仓门禁）：功能全绿、精度达标、同机 TBE 对照表、path-ledger 基线结案。未满足则禁止调 ubFactor / 双缓冲 / 多 key。

### 4.1 推荐矩阵（F 功能 / Q 精度 / B 边界 / N 负面）

**轴 A — dtype**（每个后续 shape 至少轮一圈）  
fp32、fp16、bf16（bf16 **独立** tiling key / 独立实例，禁止与 fp16 共用 `half`）。

**轴 B — 布局**

| ID | shape 特征 | 目的 |
|----|------------|------|
| S2 | rank2 `[N,C]` | 无空间维，`inner=1` |
| S3 | rank3 `[N,C,H]`（含样例 `[4,6,8]`） | 最小「像真的」 |
| S4 | NCHW | 常见训练形 |
| S5+ | 5D/6D | 高 rank 广播 |
| C1 | `C=1`，`N`/`H` 从 1 到中等 | 标量 α |
| Codd | `C` 非 32B 对齐（如 3、5、7） | 尾块 + 权重对齐 |
| Calign | `C=16/32/64` | 对齐基线 |
| N1 | `N=1` 大 `H` | 只沿 inner 切 |
| Tiny | `[1,1,1]`、`[1,1,8]`、`[2,3,1]` | 近标量、B 类候选 |
| Fat | 大 `numel`、明显能灌满全部 AIV | **唯一** A 类性能对象 |

**轴 C — 数值相**

| 相 | features | 断言重点 |
|----|----------|----------|
| 全正 | &gt;0 | `dx==grads`，`update==0` |
| 全负 | &lt;0 | `dx==w*g`，`update==f*g` |
| 含零 | 显式 0 / −0 | **零走负支的 dx** |
| 混杂 | 随机跨 0 | 泛化主路径 |
| 近 0 | ±eps（各 dtype 下溢边界） | 比较稳定性 |
| 权极端 | α∈{0,1,负,大} | 乘不炸、不误走正支 |
| 特殊值 | Inf / 可选 NaN | 与逻辑比较约定一致 |

**轴 D — 图与 Host**

- 三输入同 dtype；故意错 dtype / 错 `C` / rank1 → 失败。
- 输出两路都与 `features` 同形（**不要**期望 `update` 为 `[C]`）。
- 非连续输入（转置后再喂图）→ 与 contiguous golden 一致。
- InferShape UT：动态维、高 rank、`grads≠features` 负例。
- Tiling UT：至少覆盖 fp16 / fp32 / **bf16 不同 key**、单核 / 多核、`C=1`、尾块。

**轴 E — 精度**

- 与 `reference.py` 同策略：fp32 计算再 cast。
- 样例 json 里 fp32 `rtol/atol=1e-4` 只是起点；半精度用 Judge 默认阈值，不要自行收紧到「调不通就改阈值」。

### 4.2 基线实现顺序（和带宽视角一致）

1. Host：InferShape + 合法性（shape/dtype/rank/`C`）。  
2. Tiling：至少能表达 `outer, C, inner, usedCores, ubFactor, 尾块`；**不要死守三标量扁平字段**。  
3. Kernel：单核正确 → 多核连续切分正确 → 半精度独立类型。  
4. 填满 4.1 矩阵。  
5. **然后**才：同机 TBE 打表，只对 A 类（全核、大 `numel`）抠带宽。

在第 4 步之前改双缓冲、改 key 数量、改切核启发式，属于无效道路。

---

## 5. 性能分流：何时停，何时写仿真

### 5.1 先分类，再决定抠不抠

| 类 | 判定（本题条文） | 策略 |
|----|------------------|------|
| **A 全核** | 全部 AIV 有活、耗时明显高于 kick；任务书要 ≥ TBE **95%** | 只在这里做 GM/双缓冲/切核 |
| **B 小 shape** | 约 **&lt;10 μs**，与 TBE 差 **≤3 μs** | **停优化**，仿真图 + 分析 |
| **C 未列验收** | 奇异 rank、单核玩具例 | 记录，不立项 |

样例 `[4,6,8]`（192 点）必然是 B/C：灌不满 910B。用它调「全核 95%」是方法论错误。

### 5.2 允许做的 A 类优化（短队列，有时限）

按仓内默认：常规 tiling / 全核 / 双缓冲 **0.5～1 人日**；微秒级 **≤2 小时** 必须转 B。

A 类只允许这些假设（都能用「是否少碰 GM / 是否盖住搬运」验证）：

1. 按通道连续切分 + 权重 UB 常驻（减不规则读）。  
2. 5 路融合单趟 + 真双缓冲（tile≥2 时）。  
3. `ubFactor` 按 UB 反推并对齐 32B，减少 kick 次数。  
4. 大 shape 才 `blockDim=coreNum`，避免假全核。

增益判据：同机对照表上 A 类 case 相对上一版 **≥2%** 或明确打到 95%。否则结案写瓶颈（几乎一定是 HBM 或 kick，不是 ALU）。

### 5.3 止损触发（命中即停，24h 内写 DEC）

- 功能/精度/泛化未绿却在改性能。  
- A 类标准项到期且增益 &lt;2%。  
- 连续两条互斥路（例如「flatten gather 权重」vs「按通道扫」）都未到 95%。  
- 无 profile / 流水线证据。  
- **为 B 类已超过 2 小时**。  
- 准备引入 GM workspace、stride kernel、update reduce「加速」——方向与语义/带宽双双相反，直接回退。

### 5.4 B 类该交什么，而不是继续改代码

任务书允许「仿真证明 Ascend C 与 TBE 一致或更好」。材料应能让评审 **不必再跑一轮调参**：

1. **shape / dtype / 核数 / 耗时**：本实现 vs 同机 TBE；标出 &lt;10μs 与差值。  
2. **仿真或流水线示意**：CopyIn（3 路）/ Compute / CopyOut（2 路）；标出 kick、scalar、尾块空转。  
3. **算术强度与 roofline 一段话**：5× 流量、算力空闲；再压指令也换不回 μs。  
4. **切核反证**：若强行全核，每核元素数 &lt; 对齐粒度，预期变差（可附一张「核数↑、时延不降」）。  
5. **结论句**：B 类走豁免；A 类若已 ≥95% 则性能阶段结束。  
6. 不把样例 192 点写进「性能达标」表。

### 5.5 和模板风险的对应（避免做错方向）

| 模板风险 | 正确处置 | 错误处置 |
|----------|----------|----------|
| 空 kernel / 空 tiling | 先正确单趟实现 | 先抄一套「通用 elementwise 双缓冲」 |
| 双缓冲队列已声明 | A 类、tile≥2 再启用；UB 按 5 路重算 | 认为 BUFFER_NUM=2 已等于性能达标 |
| 扁平 tiling 字段 | **补 `C/inner/outer/尾块/核数`** | 用 `totalNum` flatten 硬写完再优化 |
| bf16→half key | **先拆类型** | 当「半精度共用路径」的性能合并 |
| InferShape 不完整 | P0 修校验与两输出同 `features` | 等性能过了再补 Host |

---

## 6. 建议工作节奏（给实现同学）

```text
InferShape + 负例
    → 带通道信息的 tiling（勿锁死三字段）
    → 单核融合 kernel（权重不落 GM）
    → 矩阵 4.1 全绿（含零值 / C=1 / 高 rank / bf16 独立）
    → 同机 TBE 表：标 A / B / C
    → 只对 A：连续切分 + 真双缓冲 + ub 反推（≤1 人日）
    → A 达标则停；B 直接仿真包，勿抠 μs
```

---

## 7. P0 / P1 / P2

### P0（不做则功能/验收必炸，禁止开性能专题）

1. **语义落地**：逻辑 `features>0`；`update` 同形不 reduce；通道维对齐广播 `[1,C,1,...]`。  
2. **InferShape / 校验**：两输出 = `features`；`grads` 同形；`weights` 为 `[C]` 且 `C=features[1]`；rank≥2；非法失败。  
3. **bf16 与 fp16 分离**：禁止 MODE_0 + `half` 兼做 bf16。  
4. **Tiling 表达通道**：不能只有 `totalNum/blockFactor/ubFactor`，否则 C=1 / 高 rank / 尾块无法稳定正确。  
5. **泛化矩阵 4.1 先绿**：全正/全负/含零、C=1、高 rank、非对齐 C、三 dtype；非连续只测图侧 AutoContiguous。  
6. **零值探针**：`features==0` 时 `dx=w*grads` 不是 `grads`。

### P1（A 类带宽，才对 95% 负责）

1. 单趟 3 读 2 写融合，**不物化权重、不写 GM 中间量、workspace 保持 0**。  
2. 按 `N/C/inner` 连续切核；大 shape 真全核；权重 UB 常驻。  
3. UB 反推 `ubFactor` + 32B 对齐；双缓冲仅在 tile≥2 的 A 类开启。  
4. 同机 TBE 对照表；只报全核 case 的 95%。  
5. A 类标准优化到期或增益 &lt;2% → DEC，不换方向盲改。

### P2（默认可不做 / 做了要能停）

1. B 类（&lt;10μs、差 3μs）：**仿真图 + 分析结论**，2 小时 profile 上限。  
2. 不为 stride、为稀疏写、为 update 预 reduce 开分支。  
3. 不为小 shape 增加 tiling key / 专用小核调度。  
4. 样例 `[4,6,8]` 只当功能烟测，不当性能门禁。  
5. C 类奇异 shape：记录，不立项。

---

**一句话**：PReluGradUpdate 在 910B 上是「5 路搬运 + 通道广播」的带宽算子；先用矩阵钉死零值、C=1、高 rank、bf16 独立类型，再用「不增加 GM」换 A 类 95%，小 shape 停手写仿真。
