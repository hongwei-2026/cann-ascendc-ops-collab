# AI-2（Grok）独立分析 — Host / InferShape / Tiling / 切分策略

| 项 | 内容 |
|----|------|
| 题名 | `SyncBatchNormGatherStats_2332` |
| 角色 | Analyst；本路视角固定为 **Host / InferShape / Tiling / 切分** |
| 代码只读 | `people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/` |
| 本文件 | `tasks/active/SyncBatchNormGatherStats_2332/analysis/AI-2_grok.md` |
| 日期 | 2026-11-17 |
| 隔离 | 不引用其他算子的门禁数字、core 切分经验或「已达标」结论 |

本文只评价现有 Host 是否把 **通道维 C、设备维 worldSize、count 张量、dtype key** 交给 Kernel 的方式合理；不改代码、不写 SUMMARY。

---

## 1. 语义与验收门禁理解（从 Host 侧）

### 1.1 算子在图上做什么

Host 看到的接口（`sync_batch_norm_gather_stats_def.cpp`）：

| 角色 | 名称 | dtype 组合（4 组） |
|------|------|-------------------|
| in0 | `total_sum` | float / float16 |
| in1 | `total_square_sum` | 与 in0 同 |
| in2 | `sample_count` | int32 或 int64（与浮点交叉成 4 组） |
| in3 / in4 | `mean` / `variance` | 与 in0 同 |
| out0 / out1 | `batch_mean` / `batch_invstd` | 与 in0 同 |
| out2 / out3 | `new_mean` / `new_variance` | **恒 float** |
| attr | `momentum=0.1`，`eps=1e-5` | float |

平台配置目前只有 `ascend910b`。全部 `AutoContiguous()`，Tiling 可以按连续 ND 读，不必为非连续 stride 开 key。

语义权威是 `samples/reference.py`（与 910B 内置核 + 图上 Cast 对齐），Host 必须服务的不变量：

1. 浮点输入先落到 **fp16 网格**，中间 **fp32**，输出再落到 fp16 网格（即便声明 float32）。
2. `n==1` 时无偏系数 `n/(n-1)` 为 **+inf**，不得在 Host 侧「保护」成有限值。
3. `total_sum` 布局约定为 **`[..., C]`，前导维折叠成 worldSize**；Kernel 沿设备维对同一通道累加。
4. `sample_count` 为标量或长度为 worldSize 的向量；**分母 n 是 count 张量上的整数和**（见 §6.4 与 golden 的差异）。

Host **不负责**做 fp16 量化；它只应：推导输出 shape、算出 C / worldSize / countNum / count 宽度、按 C 切核、把 attr 以 **bit-exact float32** 塞进 tiling。

### 1.2 验收从 Host 能覆盖什么

- **功能/精度**：`samples/case.json` 首点是 `[1,6]` + `count=[1]` int32 + C=6。这是 **B 类小 shape**：只能证明 InferShape 拷贝 `[C]`、worldSize=1、单核、`ubFactor` 对齐，**不能**当性能结论，也不能证明 worldSize>1 或 int64 count。
- **Tiling UT**：`tests/ut/op_host/test_sync_batch_norm_gather_stats_tiling.cpp` 目前只有上述 C=6 一条，期望 `"6 6 8 1 1 0 1036831949 925353388 "`（`totalNum blockFactor ubFactor worldSize countNum countIsInt64 momentumBits epsBits`）。改切分必须同步这条串。
- **小 shape 豁免**：本路 **未见到本题任务书把「<10us 差 3us」写成条文**。因此：小 C 达不到满核 **默认不当作失败**，但也不要把别的题的豁免数字抄过来当本题事实。A 类是否要求满核、是否对 TBE 95%，以本题任务书为准；任务书未进仓前，性能只记「候选」，不定案。
- **Workspace**：当前 `WS_SYS_SIZE=0`。与 Kernel 无跨核归约、无 workspace 缓冲一致。正确性路径不要为切分去加 workspace。

### 1.3 Host 必须保证、Kernel 才能对的契约

```text
totalNum     = C = total_sum 最后一维
worldSize    = 前导维乘积（rank==1 时为 1）
countNum     = sample_count 元素个数
countIsInt64 = (DT_INT64 ? 1 : 0)
blockFactor  = 每核通道数（最后一核用 remain 裁）
ubFactor     = 单次 UB 通道数（Kernel 再与 blockLength 取 min）
SetBlockDim  = usedCores = ceil(C / blockFactor)
tilingKey    = fp16 → schMode 0；fp32 → schMode 1
momentum/eps = float 位型，禁止 Host 再量化
```

Kernel 按 `offset = device * channels_ + channel` 扫 GM。因此 **C 必须是最后一维且与 mean 的通道数一致**；Host 现在几乎不校验这一点（§6）。

---

## 2. 现有 tiling.cpp 是否合理（逐项）

### 2.1 worldSize / channels

```text
rank<=0 → C=1, worldSize=1
rank==1 → C=dim0, worldSize=1
rank>=2 → C=最后一维，worldSize=∏前导维
```

**合理处**：与 Kernel 的 `[worldSize, C]` 线性布局一致；`[1,C]` 样例得到 worldSize=1；多 rank 前导维折叠（例如误标成 `[G,N,C]`）仍能按连续元素当设备维扫完，避免漏加。

**问题**：

| 点 | 说明 |
|----|------|
| 不读 square_sum | 未断言与 `total_sum` 同 shape；错 shape 会在 Kernel 越读而 Host 仍 SUCCESS |
| 不读 mean/var 元素数 | InferShape 跟 mean；Tiling 跟 sum 的 C。若 `mean.numel()!=C`，图能过、核错位 |
| 0 通道 | `channels<0` 才失败；C=0 会 `usedCores=1` 且 Kernel 早退，行为合法但无 UT |
| int64 乘积 | 前导维很大时 `worldSize *= dim` 可能溢出；BN 实际 N 很小，P0 可 squawk 不必改公式 |

**结论**：折叠策略 **采纳为 P0**；补校验是正确性加固，不是性能。

### 2.2 countNum / int32 vs int64

- `countNum = sample_count.GetShapeSize()`，负值钳成 0。
- `countIsInt64` 仅看 `DT_INT64`，**其它类型一律走 int32 解释**。
- OpDef 只有 int32/int64，非法 dtype 理论上进不来；若图上出现其它整数，会静默按 int32 读，属 Host 纵容。

Kernel 对 `countNum` 个元素 **求和** 得到 n。Host 不把 n 算死进 tiling（count 是运行时 GM），这是对的：同一 shape 不同 count 不必重 tiling。

**不要**为 int32/int64 再拆一套 tiling key：宽度已经是 data 字段，Kernel 分支廉价，key 翻倍只涨编译。

### 2.3 多核与 `kMinBlock=8`

现逻辑：

1. `usedCores = min(AIV核数, C)`（C=0 时 1）。
2. `blockFactor = ceil(C / usedCores)`。
3. 若 `blockFactor < 8`：抬到 `min(C, 8)`，再 `usedCores = ceil(C / blockFactor)`。
4. 再算一遍 `usedCores = ceil(C / blockFactor)`。

意图正确：**不要把单通道（短向量 + 启动开销）撒到 48/64 核**。fp32 一个 32B 块正好 8 个通道。

数值例子（只描述公式，不假设某 SoC 核数）：

| C | 若 AIV=48 | 结果 |
|---|-----------|------|
| 6 | min-block 触发 | 1 核，blockFactor=6（与 UT 一致） |
| 8 | 抬到 8 | 1 核 |
| 16 | | 2 核 × 8 |
| 64 | blockFactor 先为 2<8 | **8 核 × 8**，不是 48 核 |
| ≥ 8×AIV | 不触发 min-block | 可满核 |

因此：**C 不够大时，本切分故意不满核。** 若任务书 A 类 case 的 C 只有几十，满核优化从 Host 切不开，应走小 shape / 仿真，而不是把 `kMinBlock` 降到 1。

`kMinBlock=8`：**P0 保留**。只有 A 类实测证明「核太少是主因」才允许在台账里试 4 或 16（一次改一个数）。

### 2.4 ubFactor

```text
budget = ubSize - 4KiB（过小则 ubSize/2）
maxTile = align_down(budget / (7 * 4字节), 8)，钳在 [64, 8192]
ubFactor = min(blockFactor, maxTile)，再向上对齐到 8，不超过 maxTile
```

- `kBufCount=7` 与 Kernel 七块 `TBuf<float>` 一致，Host/Kernel 预算同源，**合理**。
- 上限 8192：大 C 靠多轮 UB 循环，避免单核一次申请过大；910B UB 通常远大于 `7*8192*4`，此帽偏保守，P0 可留。
- **对齐后 ubFactor 可以 > 本核实际通道数**（C=6 → ubFactor=8）。Kernel `Process` 用 `min(remain, ubFactor)`，Load 带 pad，**不会按 8 写回 C=6**。Host 不必把 ubFactor 再压回 blockFactor；UT 已钉死 8。

风险：若有人改 Kernel 按 `ubFactor` 而不是 `len` 写 GM，尾核会写穿。那是 Kernel 契约，Host 侧在注释/UT 里钉 `ubFactor>=len` 即可。

### 2.5 dtype tiling key

```text
DT_FLOAT16 → GET_TPL_TILING_KEY(MODE_0)  → Kernel half
DT_FLOAT   → MODE_1                       → Kernel float
其它       → GRAPH_FAILED
```

与 `tiling_key.h` 的 `schMode` 两档、`sync_batch_norm_gather_stats.cpp` 的 `if constexpr` 一致。

**刻意不进 key 的东西**（本路建议维持）：

- count int32/int64 → `countIsInt64` 字段
- worldSize=1 vs >1 → 运行时循环 1 次或 N 次
- count==1 出 inf → 运行时，禁止 Host 特判改公式

**不要**为「fp32 输入但按 fp16 语义」再开第三 key：量化在 Kernel，Host 只按 **声明 dtype** 选加载宽度。

### 2.6 Attr 位型

`FloatToBits(momentum/eps)` 写入 int64 字段，Kernel `BitsToFloat` 还原。避免 Host/Device 各 parse 一遍 attr 造成 0.1 的二进制差。缺 attr 时默认 0.1 / 1e-5，与 OpDef 一致。acl 样例用 double 传入，依赖 GE 转 float；Host 不要再做 double 计算。

### 2.7 InferShape

四份输出 **全部 `*meanShape`（input 3）**。

与样例一致：mean 是 `[C]`，四输出 `[C]`。`new_mean/new_var` 虽 dtype 恒 float，shape 仍是逐通道。

缺口：

- 无 `InferDataType` 实现：依赖 OpDef 四组绑定。一般够用；若图把 fp16 输入接到 float 输出槽，不是 Tiling 能救的。
- 不校验 `mean` 最后一维 == `total_sum` 最后一维。
- 若 mean 被做成 `[1,C]`，输出也会 `[1,C]`，而样例/判题常见 `[C]`。Host 不应擅自 squeeze；应在设计里写明 **mean 与输出 rank 一致**。

---

## 3. 最简正确实现路径（Host 视角 P0）

目标：图能过、切分与现 Kernel 契约一致、覆盖 4 组 dtype 与 count 宽度；**不改切分公式、不加 workspace、不增 tiling key**。

1. **冻结当前切分**：C 最后一维、前导维积为 worldSize、`kMinBlock=8`、ub 七缓冲、key 仅 fp16/fp32。这已能跑通 `[1,6]` UT。
2. **正确性只读摸底**（实现阶段、非本分析改代码）：云 `-e` 对照 `reference.py`；再补 Host 能构造的边界 tiling 用例（不必先改 Kernel）：
   - rank-1 `total_sum=[C]` → worldSize=1
   - `[N,C]` N>1 → worldSize=N
   - `sample_count` int64、`countNum=1` 与 `countNum=N`
   - C=1 / C=8 / C=9（9 触发 2 核：8+1）
   - 非法浮点 dtype 失败；momentum/eps 位型与默认 attr
3. **InferShape 保持「拷贝 mean」**，直到出现 mean 与 C 不一致的官方用例再加校验失败，而不是 Host 重算 shape。
4. **count 语义跟 Kernel：求和**。若某条官方 golden 只取 `count[0]`，再单独立项，禁止 P0 在 Host 把 `countNum` 改成 1 去「迎合」`reference.py` 的 flatten。
5. Workspace 保持 0。attr 只传 bits。

P0 完成定义：Tiling UT 绿、首点 `-e` 与 reference 对齐、四组 dtype 至少各一条 Host key/字段正确。满核与 ub 调参不在 P0。

---

## 4. 性能候选路径（有序、有止损）

前提：功能精度过线，且有 **本题** 任务书 A 类 shape / 对照算子。未满足则下列只是排队，不执行。

| 序 | 候选 | 假设 | 时限 | 止损 |
|----|------|------|------|------|
| P1-a | **维持 min-block=8，只保证大 C 满核** | A 类 C ≥ 8×AIV 时 usedCores 已满，瓶颈不在 Host | 对照表 0.5 人日 | 已满核仍差门禁 → 交给 Kernel（向量/流水），**不要再切碎 C** |
| P1-b | **ubFactor 帽 8192** | 大 C 多轮循环 vs 更大 tile | 半日 + 1 组大 C profile | 增益 <2% 或 UB 不够 → 回 8192 |
| P1-c | **kMinBlock 敏感度（4 或 16）** | 中等 C（例如数十～数百）核数不足或过碎 | 一次只改一个常数，半日 | 小 C 变慢或精度无关回归 → 回 8；**禁止为样例 C=6 调 min-block** |
| P2 | **沿 worldSize 切核 + workspace 归约** | 仅当 C 极小且 N 极大、且任务书点名该 shape | 须 DEC；≤2 人日 | 需要核间 add，复杂度高；N 在 SyncBN 通常很小 → **默认不立项** |
| P2 | **count 预归约进 tiling / 标量 Host 算 n** | 省掉每核 `GetValue` 循环 | 不值得：countNum≪C | 直接废弃除非 count 向量极长 |

分类提醒：

- **A 类**：任务书列出的大 C / 全核 case → 才值得动 Host 切分。
- **B 类**：C=6 及「核启动 >> 计算」→ 仿真说明，不抠 Host。
- **C 类**：未列 shape、畸形 rank → 记录，不优化。

---

## 5. 「不要先做」反模式

1. **不要先为性能改 tiling key 矩阵**（count 宽度、worldSize=1、是否 inf、是否双缓冲）。现在 2 key 已经对齐 Kernel 模板。
2. **不要先把 C=6 撒到满 AIV**。min-block 存在就是为了避免这件事。
3. **不要先按 `reference.py` 把 `total_sum` reshape(-1) 当纯 `[C]`、把 worldSize 恒置 1。** 那是样例 N=1 的等价写法，不是 Host 合同。
4. **不要先在 InferShape 里用 `total_sum` 推输出而丢掉 mean 的 rank。** 输出与 running mean 对齐是接口语义。
5. **不要先加 workspace 做设备维归约或 count 广播。**
6. **不要先做 Cube / ND2NZ / 特殊 Format。** OpDef 只有 ND。
7. **不要先把 ubFactor 强行等于 blockFactor** 去「好看」：会破坏 8 对齐与现 UT。
8. **不要先在 Host 把 n==1 改成有限无偏、或对 momentum/eps 做 fp16 量化。**
9. **不要串题套用其它算子的 min-block、满核百分比、豁免 us。**
10. **不要在分析阶段改 Kernel 去迁就尚未证实的 Host 新切分。**

---

## 6. 本题特有风险

### 6.1 Shape 推导（InferShape vs Tiling 两套源）

| 字段 | 源 |
|------|----|
| 四输出 shape | **mean** |
| C、worldSize、核切分 | **total_sum** |

不一致时：图 shape 合法，Kernel 按错误 C 索引。P0 应用例：`sum=[1,C]` vs `mean=[C]`（通过）；`mean` 多一维（记录，不擅自 squeeze）。缺校验不是立刻改公式，而是 UT 钉期望。

### 6.2 world_size 折叠

前导维全部乘进 `worldSize`，Kernel 做 `N` 次长度为 `len` 的连续读。

- **N=1（当前样例）**：循环一次，与 flatten 数值等价。
- **N>1**：必须沿设备维 **加和再 / n**。若有人按 reference 把 `[N,C]` flatten 成 `N*C` 个通道，Host 的 C 取最后一维会与 golden 人数对不上——那是 **golden 误用**，不是切分 bug。
- **不要**把 worldSize 再拆到多核：通道已经独立，设备维是归约轴。

### 6.3 小 C 分核

`kMinBlock=8` 使 C<8×AIV 时 **故意少核**。样例 C=6 单核是设计结果。风险是评审误判「没用满核」。应对：用公式说明满核阈值，而不是降 min-block 刷核数。

尾核 `blockLength < blockFactor`（如 C=9 → 8+1）依赖 Kernel remain 裁剪；Host 的 `SetBlockDim(ceil(C/blockFactor))` 不要改成「最后一个核也按 blockFactor 写」。

### 6.4 int32 / int64 count

三处可能打架：

| 来源 | 行为 |
|------|------|
| OpDef | 两组 count dtype |
| `reference.py` | 注释写 int32；实现 **只取 `reshape(-1)[0]`**，且先转 float64 |
| Kernel | **对 countNum 全元素求和**；int32 先扩到 int64 再加，再转 float |

Host 当前把 `countNum` 和 `countIsInt64` 如实传下去，站在 Kernel 合同一侧。

风险：

- 官方若 `sample_count` 为 `[world_size]` 且语义是 **每卡局部计数、应求和**，现 Host/Kernel 对、reference 错（只取第一个）。
- 官方若语义是 **已经是全局 n、只读第一个**，则 Host 不应在 N>1 时仍传 `countNum=N` 让 Kernel 加出 N 倍。
- int64 大计数转 float32 会丢整数精度（>2^24）。这是 Kernel/语义问题；Host 不要提前把 count 压成 float tiling 字段。
- `countNum=0` 时 Kernel 得到 n=0，随后 `/n`。Host 把负 shape size 变成 0，可能把非法输入变成除零。P0 可对 `countNum<=0` 打日志或失败，但不要静默改成 1。

**本路建议**：P0 保持「countNum=numel、求和」；用至少一条 N>1 且 count 向量的用例向任务书/内置算子对齐；在对齐前 **禁止** Host 把 countNum 折叠成 1。

### 6.5 其它 Host 风险（次要）

- 未校验 `total_square_sum` 与 `total_sum` 同形。
- `GetInputShape(0)` 空指针有检查；attrs 空则用默认——好。
- 仅 910B：不要为其它芯片写 key。
- TilingParse 空实现：无 compile info 缓存，可接受。

---

## 7. 废弃建议清单（本路否定，无新证据不得重开）

| ID | 方向 | 原因 |
|----|------|------|
| D1 | 按通道 1:1 分满 AIV（弃 min-block） | 短向量+启动；与现 UT/注释相反 |
| D2 | worldSize 切核 + workspace 跨核 Add | 归约轴切开会引入同步；N 通常小 |
| D3 | count 宽度 / n==1 / N==1 做成 tiling key | 字段已够；key 膨胀 |
| D4 | Host 预计算 n 或预量化 momentum/eps | 破坏动态 count 与 bit-exact attr |
| D5 | InferShape 改为「flatten total_sum」或 squeeze 成 1D | 与 running mean 接口不符 |
| D6 | P0 把 countNum 恒置 1 以贴合 reference 的 `[0]` | 未证实官方 N>1 语义前会埋雷 |
| D7 | 为 C=6 样例做 Host 性能专项 | B 类；非门禁 |
| D8 | Cube、私有格式、非 ND | OpDef 无此需求 |
| D9 | 用其它题目的满核百分比 / us 豁免当本题门禁 | 串题 |

允许以后重开的 **仅**：任务书明确 A 类 shape 后的 P1-c（min-block 4/16 二选一实验），必须新 path-ledger 行。

---

## 8. 本路结论（给 SUMMARY 用，本文不代写 SUMMARY）

- **基线 Host 可采纳**：现 `tiling.cpp` 的 C/worldSize 折叠、`kMinBlock=8`、ub 七缓冲与 8 对齐、两档 dtype key、count 字段化、workspace=0，与现 Kernel 契约匹配，P0 **不要重写切分**。
- **P0 Host 工作**应是校验缺口与 UT 补全（N>1、int64 count、C=1/8/9、非法 dtype），而不是调核数。
- **性能**：先看 A 类 C 是否达到 `8×AIV`；达不到就不要从 Host 逼满核。
- **最大语义坑**在 count：reference 取首元、Kernel 求和——Host 保持透传 numel，等官方用例裁决，禁止抢先折叠。

下一闸（实现阶段，非本文件）：用户确认后云 `-e`；改 tiling 公式前先更新 Host UT 期望串。
