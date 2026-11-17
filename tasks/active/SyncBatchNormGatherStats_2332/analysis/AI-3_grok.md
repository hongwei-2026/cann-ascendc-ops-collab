# AI-3 — Kernel 流水线 / 性能候选与止损

| 项 | 内容 |
|----|------|
| 题名 | `SyncBatchNormGatherStats_2332` |
| 视角 | Kernel 流水线、性能候选、止损与小 shape 豁免（本路只评这一面） |
| 代码（只读） | `people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/` |
| 日期 | 2026-11-17 |
| 阶段 | 分析。云测未跑（`STATUS.md`）。性能专题未开门 |
| 隔离 | 只讨论本题。PRelu / QuantUpdateScatter / ConfusionTranspose 的门禁数字、双缓冲结论、仿真豁免文案都不是本题事实 |

本文件不改 Kernel/Host，不写 `SUMMARY.md`。下列候选是给汇总裁决用的顺序，不是开工单。

---

## 1. 语义与验收门禁理解

权威语义是 `samples/reference.py` 头注释，以及与之对齐的 kernel 头注释：

1. 浮点输入先落到 fp16 网格（fp32 输入在核内 `CAST_RINT`→fp16→fp32；fp16 输入加载后再 `CAST_NONE` 到 fp32）。
2. 中间用 fp32：`batch_mean = Σsum / n`，`batch_var = Σsq / n - mean²`，`invstd = 1/sqrt(var+eps)`，再做 EMA。`eps` 以 fp32 参与，不回退成 fp16。
3. 四个输出再按 fp16 舍入。`batch_mean` / `batch_invstd` 按输入 dtype 写出；`new_mean` / `new_variance` 恒为 float（OpDef 四组 dtype 都如此）。
4. `n == 1` 时无偏系数是 `+inf`，允许 `new_var` 出现 ±inf/NaN。禁止改成有限数。
5. `total_sum` / `total_square_sum` 布局是 `[world_size, C]`（rank≥2 时最后一维为 C，其余维乘成 `worldSize`）。沿设备维累加，再按通道写出。

精度对照目标（reference 自述，尚未在本仓云上复核）：相对 910B 内置，50 条用例四个输出 ≤1 个 fp16 ulp；NaN/±inf 由比对器按「不超差」处理。`Test_050` 类点是量化铁律的代表：`total_square_sum=100.001` 量化成 `100.0` 后 `batch_var=0`，`invstd=1/sqrt(eps)=10000`；跳过量化会得到约 `315.86`。

### 1.1 性能门禁现在能不能用

`docs/collaboration/02-perf-gate.md` 要求四条同时成立才允许开性能专题：

| # | 条件 | 本题现状 |
|---|------|----------|
| 1 | 功能用例全绿（含约定泛化） | 未跑。云 `-e` 未做，`path-ledger` S002 仍是「规划中」 |
| 2 | 精度达到任务书 / Judge 阈值 | 未证明。代码意图对齐 reference，没有同机 bin 结论 |
| 3 | 同机 TBE（或任务书标杆）对照表 | 仓内没有本题性能表，也没有 TBE 数字 |
| 4 | 台账基线行结案为「采纳（功能精度）」 | 只有 S001「采纳（开题）」 |

四条缺一，本路所有性能候选全部挂起。先做的只有正确性冒烟，不是流水线改造。

`02-perf-gate.md` 里的「全核 ≥ TBE 95%」和「10us 以下相差 3us」写明是 **PReluGradUpdate 任务书示例**，并要求「按任务书改数字」。本题任务书性能表未入库（`STATUS`：合入仓待核对）。在抄齐本题条文之前：

- 不把 95% / 10us / 3us 写成 `SyncBatchNormGatherStats` 的验收线；
- 分类方法可以沿用：A = 任务书点名、且全核参与的性能 case；B = 任务书若给出小 shape 豁免条款则走仿真图+分析；C = 未列入验收表，记录即可；
- 豁免条款的具体阈值以将来抄入本题 `materials/` 的任务书为准。没有条款就不要自己发明豁免百分比。

### 1.2 samples 首点不是性能结论

`samples/case.json` 的 `npu_cases[0]`：

| 张量 | shape | dtype | 规模 |
|------|-------|-------|------|
| `total_sum` / `total_square_sum` | `[1, 6]` | fp32 | 6 元素 / 24 B |
| `sample_count` | `[1]` | int32 | 值 `100` |
| `mean` / `variance` | `[6]` | fp32 | 单核可装下 |

对应 tiling（`kMinBlock = 8`）：`C=6` 时 `blockFactor=6`、`usedCores=1`、`ubFactor` 对齐到 8，核内 `len=6`。向量按 8 元素对齐，有效车道 6/8。`worldSize=1`，设备循环只转一圈。

这个点只够做正确性冒烟。它的耗时会被 kernel 启动、标量同步和短向量发射主导，字节数可以忽略。禁止用它的 us 数判断「优化有没有效」，也禁止为它改双缓冲或改切分。

---

## 2. 现有 Kernel 流水线（只评 `kernel.h`）

入口 `sync_batch_norm_gather_stats.cpp` 按 tiling key 实例化 `half` 或 `float`，没有第二套 schedule。UB 固定 7 块，与 tiling 的 `kBufCount = 7` 一致：`raw / load / accSum / accSq / mean / var / tmp`，每块 `bufElems * 4` 字节。`maxTile` 被夹在 `[64, 8192]` 且 32B 对齐。

`Process` 每核做一次 `SumSampleCount()`，再按 `ubFactor` 串行调用 `ComputeTile`。每 tile 结束有一次 `MTE3_MTE2` 全等。当 `blockFactor ≤ ubFactor`（小 C、以及多数「一核一块」切分）时，这个 while 只跑一轮，**跨 tile 双缓冲的重叠窗口是 0**。

### 2.1 `ComputeTile` 实际顺序（全串行）

```text
Duplicate accSum/accSq = 0
for device in worldSize:                 # 串行，无预取
    LoadQuantized(sum) → Add             # 内部 V_MTE2 等 + MTE2_V 等 + 量化 Cast
    LoadQuantized(sq)  → Add
LoadQuantized(mean)
LoadQuantized(var)
fp32 向量链：Muls/Mul/Sub/Adds/Sqrt/Div/EMA
×4 QuantFp16Vec（每路 2 次 Cast，复用 rawBuf_）
V_S 等 → 标量 for i in len: GetValue/SetValue 补 NaN/Inf → S_V 等
V_MTE3 等
Store batch_mean / batch_invstd / new_mean / new_var   # 四次分开搬
MTE3_MTE2 等
```

`LoadQuantized` 每次都是「立刻 SetFlag + 立刻 WaitFlag」：

- 先 `Duplicate(dst, 0)`，再 `V_MTE2` 等，再 `DataCopyPad`，再 `MTE2_V` 等，然后才量化。
- fp32：`QuantFp16Vec` = `CAST_RINT` 到 half + `CAST_NONE` 回 float，写回同一 `dst`，half 落在 `rawBuf_`。
- fp16：拷进 `rawBuf_` 的 half，再 `CAST_NONE` 到 float。

等待紧贴发射，MTE2 与 Vector 没有重叠。`worldSize = W` 时，每 tile 至少 `2W+2` 次这种往返（sum、sq 各 W 次，再加 mean、var），外加 4 次写出。

### 2.2 标量补 NaN 环

向量 `Sqrt` + `Div` 之后、写出之前：

1. 四个结果先 `QuantFp16Vec`（注释：CPU 模拟上 `Sqrt(负)→0`，且 Cast 会吞 NaN）。
2. `V_S` 排空后，对 `i ∈ [0, len)` 用 `GetValue/SetValue`：`var+eps < 0` 写 quiet NaN；`var+eps == 0` 且当前 invstd 为 0 则写 `+inf`。
3. `S_V` 排空后才允许 MTE3。

这条环的固定成本是两次流水排空，与 `len` 无关；可变成本是 `O(len)` 次标量访 UB。`len=6` 时排空本身就比补洞贵。`len` 到上千时，标量环才会变成可见占比。补洞必须留在**最后一次会吞 NaN 的 Cast 之后**（见 §6.1）。

### 2.3 双缓冲现在插不进去的原因

| 事实 | 后果 |
|------|------|
| 7 块 UB 里 `rawBuf_` 被 fp16 加载、`QuantFp16Vec`、fp16 `StoreAsInputDtype` 三处复用 | 乒乓必须先拆 buffer，不能在 `rawBuf_` 仍被量化/写出占用时预取下一拍 |
| tiling 用 `kBufCount=7` 算 `maxTile` | 加乒乓却不改 Host 预算会 UB 溢出；改了预算，`ubFactor` 大约按 7→9 缩小两成 |
| 通道维已按核切开，核内常是单 tile | 跨 tile 双缓冲没有下一 tile 可藏 |
| 算术强度极低：每通道大约几次 Muls/Add、一次 Sqrt/Div | 能藏住 DMA 的只有「下一设备的拷贝 vs 当前 Add/Cast」。`W=1` 时设备循环没有第二拍 |
| 样本与 reference 注释中的已核对用例是 `[1, C]` | 跨设备流水线缺少已知的大 `W` 性能点 |

因此双缓冲是**有条件的后位候选**，不是基线下一步。

### 2.4 已经做对、不该拆掉的切分

`tiling.cpp` 把每核通道下限设为 8（不足则并核）。注释写明单通道分核会变成极短向量且开销更大。`C=6` 因此是 1 核。这和「通道独立、设备维才是归约」一致：并行维是 C，不是 world。性能阶段不要把这条改回「一通道一核」。

---

## 3. 最简正确实现路径

### 3.1 基线算不算「最简正确实现」

现有 kernel **在结构上就是最简正确实现**，不需要为了「先跑通」再写一版：

- dtype 两套模板（fp16 / fp32）+ `sample_count` 的 int32/int64 标量累加；
- 输入 fp16 量化、中间 fp32、输出再量化；
- `n==1` → `PositiveInf()` 再 `Muls`；
- 负 `var+eps` 在量化之后标量补 NaN；
- 通道多核、核内单缓冲顺序执行。

它还不是「正确性已达标」：

- 没有云上 `-e` 全文 log，没有与 `reference.py` / 内置 bin 的同机对照；
- S002 未结案。在 S002 记成「采纳（功能精度）」之前，任何流水线改动都不算优化。

P0（本路认可、且先于全部 K 候选）只有：

1. 云 `build.sh -j8 -e` 冒烟，失败 log 归档到本题 `explorations/logs/`。
2. 对照 reference / 内置：普通点、`count==1`、负方差补 NaN、`Test_050` 类 fp16 量化（`100.001→100`）、四组 dtype。
3. 台账把功能精度行结案。全绿再抄任务书性能表，做同机 TBE 对照。对照之前不改 `ComputeTile` 的事件和 buffer。

### 3.2 和 reference 还没对齐的语义差（正确性，不是性能）

这些差不解决，就不能用「reference 很简单所以循环可以删」来做性能：

| 点 | Kernel | `reference.py` | 影响 |
|----|--------|----------------|------|
| 设备维 | `for device` 累加 `offset = device * C + channel` | `reshape(-1)` 后逐元素除 `n`，不沿设备归约 | 仅 `world=1` 时与核一致。已入库样本是 `[1,6]`。`W>1` 时两边不是同一公式 |
| `sample_count` | `SumSampleCount` 把 `countNum` 个 int32/int64 **全加** | 只取 `reshape(-1)[0]` | 多元素 count 时 n 不同。性能上这段每核只跑一次，不是热点 |
| `new_var` 无偏项 | fp32 `Muls(accSq, unbias*momentum)` | `float64` 的 `n/(n-1) * mom * var` 再收成 fp32 | `n==1` 的 inf×0、大 n 的 ulp 要以 bin 为准 |
| fp16 的 `batch_invstd` 写出 | 量化并标量补 NaN 之后，`StoreAsInputDtype<half>` **再** `CAST_RINT` | `_q16` 一次 | 第二次 Cast 有机会再吞掉刚补上的 NaN。见 §6.1 |

P0 用用例把这四行钉死。钉死之前不要删 world 循环，不要改 count 的归约，不要把补 NaN 挪到 store Cast 前面。

---

## 4. 性能候选路径（有序、有止损、DEC 条件）

总闸：§1.1 四条件 + 本题任务书性能表已抄入。未过闸则下表全部不写代码。

过闸后的第一件事仍不是改 kernel：先出同机 TBE 对照，再按任务书把 case 分成 A/B/C。

- 若 A 类已经达到任务书条文：写报告，性能阶段结束。
- 若任务书把小 shape 列为可豁免，且实测落在该条款内：仿真图 + 分析，停止抠 us。`C=6` 只可能落在这类或「非门禁」，不可能单独证明一条优化。
- 只有 A 类未达标，才按 K1→K5 各开 **一条** `path-ledger` 行。命中止损（时限到且增益 &lt;2%、或无 profile 证据）后 24h 内写 `decisions/DECxxx.md`，未写 DEC 不得开下一条。

默认时限对齐 `02-perf-gate.md`：标准项 0.5～1 人日；结构性大改须书面立项且 ≤2 人日；小 shape 微秒级 ≤2 小时后转豁免材料。

### K0 — 只测量（0.5 人日，不改 kernel）

同机跑任务书 A 类（及任务书列出的小 shape），记录 TBE、本核、核数、`worldSize`、`C`、是否单 tile。

**停：** 没有 A 类差距就停。对照表本身是后续所有 DEC 的输入。禁止用 `case.json` 的 `[1,6]` 填这张表。

### K1 — 热循环事件从「申请完立刻等」改成可重叠的固定 event（≤1 人日）

现状每条 `DataCopyPad` / fp16 store 都 `FetchEventID` 后马上 `WaitFlag`，重叠为 0。候选是固定 event 对，在不破坏 RAW 的前提下，让下一次 MTE2 与当前 Cast/Add 重叠。

**只在** K0 的 profile 显示 A 类 stall 在 `MTE2_V` / `V_MTE2` 时开工。

**止损：** 增益 &lt;2%，或 stall 不在 MTE 而在启动/标量，则 DEC 停止。

**正确性约束：** `rawBuf_` 三处复用（§2.3）。重叠窗口不得让下一拍 DMA 覆盖仍被 `QuantFp16Vec` 或 fp16 store Cast 使用的 half。fp32 输入的量化写 `rawBuf_`，fp16 加载也写 `rawBuf_`，先拆一块专用 half buffer 再谈重叠。

### K2 — worldSize 软流水（仅当 A 类里 W≫1）

乒乓 `load`（以及 fp16 的 half 落地），设备 `i+1` 的 `DataCopyPad` 盖住设备 `i` 的 Add+量化。

**开工条件（同时满足）：** K0 证明 A 类未达标；profile 把缺口主要记在设备循环的 MTE2 等待；任务书性能点的 `worldSize` 大于 1。仓内没有这样的点之前，本条保持关闭。

**止损：** ≤1 人日。若只在 `W=1` 上有收益，或收益 &lt;2%，废弃。`W=1` 时循环体仍在，但没有第二台设备可预取，mean/var 两次加载之间只有一条 `Add`，藏不住一轮 GM 延迟。

**DEC：** K1 之后 A 类仍失败，且 stall 占比仍落在设备循环，才允许把 K2 写成新假设。不允许和 K1 同一次提交混改。

### K3 — 标量 V_S 环改向量谓词（仅大 tile 且 profile 证明）

用 Compare/Select 在**最后一次吞 NaN 的 Cast 之后**写 quiet NaN / `+inf`，去掉 `O(len)` 的 `GetValue/SetValue` 和成对的 `V_S`+`S_V`。

**开工条件：** A 类 `len` 达到数百以上，且 profile 里标量环或这两次排空占比明显。`len=6` 或 `ubFactor` 只有几十时不做——两次排空相对启动可以忽略，向量 Compare 还有 NaN 载荷风险。

**止损：** ≤0.5 人日。必须单列用例：`var+eps<0`、`var+eps==0`、`count==1`（inf 路径不要和补 NaN 搅在一次改动里）。任一 bin 偏离 reference，整条回退到标量环。

**DEC：** 若 A 类差距主要在标量环，且 K3 回退后没有 bit 级等价的向量写法，结论写「保留标量环」，不再换第三种补洞。

### K4 — 去掉多余的整段 Duplicate，合并写出侧同步（≤0.5 人日）

`LoadQuantized` 每次整段 `Duplicate(0)` 再 Pad。Pad 已带 0 填充时，对齐良好的整 tile 上这次 Duplicate 可能多余；尾段 `[len, calcCount)` 仍要是 0，因为 `Add/Muls` 用的是 `calcCount` 而量化/写出用的是 `len`。只能在 `len` 已是对齐长度、且 Pad 覆盖尾元素时删。

fp32 的四次 `StoreFloat` / `StoreAsInputDtype<float>` 之间没有逐次 `MTE3_V`（buffer 各不相同，可以排队）。fp16 的 `batch_*` 两次写出共用 `rawBuf_`，store 内部还有 `V_MTE3` 等，这是别名造成的串行，不是漏写的双缓冲。单独为 fp16 再加一块 half 只值当 K0 显示 MTE3 占比高。

**止损：** 收益 &lt;2% 或只出现在 B 类，停止。禁止为了少一次 Cast 删掉输入/输出 fp16 量化。

### K5 — 整 tile 双缓冲或按 world 切核（结构性，默认不开）

跨 tile 双缓冲：加 2～3 块 UB，`kBufCount` 与 `maxTile` 一起改。单 tile/核时重叠为 0（§2.3），这条天然无效。

按 world 切核再 workspace 归约：通道维才是独立并行维；设备维是短归约。多一次 GM 写回 + 核间同步，在 `W` 为个位数时通常慢于核内 `Add`。只有任务书出现「`W` 很大且 `C` 小到并核后仍是单核在扫设备」才值得立项。

**开工：** K1～K4 的 DEC 已写明 A 类仍失败，且新假设指向「单 tile 内设备循环」或「短 C + 极大 W」二者之一。时限 ≤2 人日。到期三选一：继续（必须是新假设）、回退到 K1 之前的功能基线、或按任务书走仿真豁免。

### 4.1 候选和 shape 的关系（避免做错对象）

| 形态 | 流水线事实 | 允许的动作 |
|------|------------|------------|
| `W=1`，`C` 很小（样本 `[1,6]`） | 1 核、1 tile、设备循环 1 次、向量不满 | 只做正确性。耗时不进性能结论 |
| `W=1`，`C` 大到多核且每核多 tile 或单 tile 但向量拉满 | 跨设备没有第二拍；跨 tile 才可能有 MTE/Vector 重叠 | K0 后视 A 类再考虑 K1/K4。K2 关闭 |
| `W>1`，`C` 足够分核 | 核内设备循环是主要额外带宽 | K0 证明后再开 K2 |
| `W` 极大、`C` 很小 | 通道并行用不满，单核扫设备 | 先确认这是不是任务书 A 类；是才允许讨论 K5 |

没有任务书 shape 表时，上表只是分流规则，不能当成「大 C 一定要做」。

---

## 5. 「不要先做」反模式

1. **跳过功能精度基线改流水线。** 四条件未齐就改 event、加 buffer、改 world 循环，一律不算性能工作。
2. **用 samples 首点 `C=6` / `[1,6]` / `count=100` 下性能结论**，或为这个点调 `ubFactor`、核数、双缓冲。
3. **把 PRelu 的 95%、10us、3us，或其他题的双缓冲/全核经验抄成题。** 那些题的 EXP 不能当本题 profile。
4. **为省 Cast 去掉 fp16 量化**，或给 fp32 做一条「不量化」快路径。`Test_050` 会从 10000 变成约 315.86。
5. **把 `count==1` 的 inf 收成大数或 0。** 无偏系数是语义的一部分。
6. **在 store 侧最后一次 Cast 之前用向量写法替换标量补 NaN**，或把补 NaN 移到 `QuantFp16Vec` 之前。Cast 吞 NaN 是这条环存在的原因。
7. **删掉 `worldSize` 循环**，理由是 reference 的 `reshape(-1)` 没有归约。那只在 `W=1` 的已核对用例上成立。
8. **一通道一核**，或把 `kMinBlock=8` 改掉去「提高占用」。短向量 + 启动会更差，tiling 注释已经写了这个方向。
9. **上 Cube / SIMT / 多核 atomic 归约通道。** 计算是逐通道归约加 EMA，并行维已经是 C。
10. **把 count 的标量 `GetValue` 循环向量化。** 每核一次，且「全加」还是「只取 `[0]`」尚未用用例钉死。
11. **K1 与 K2、K3 捆在同一次改动里。** 增益无法归因，回退会把正确性补丁一起撤掉。
12. **小 shape 上连续抠超过 2 小时仍不写 DEC。** 到时限就改出仿真/分析材料（若任务书有豁免条款）或明确「非门禁、不优化」。

---

## 6. 本题特有风险

### 6.1 fp16 量化是语义税，也是流水线别名

fp32 输入每加载一块就要两次 Cast（`rawBuf_`）。每 tile 大约 `(2W+2)` 次加载量化，输出再 4 次 `QuantFp16Vec`。这些 Cast 不能删。

fp16 输入少一次「加载时 RINT」，但 `batch_mean` / `batch_invstd` 在 `QuantFp16Vec` 之后，`StoreAsInputDtype<half>` 还会再 `CAST_RINT` 到同一块 `rawBuf_`。`new_*` 走 `StoreFloat`，不再 Cast。

因此：

- 任何「融合量化与写出、少一次 Cast」都要先证明 fp16 网格值不变，并且 **NaN/Inf 还在**；
- `rawBuf_` 是加载、量化、fp16 写出的共享资源。双缓冲若只复制 `loadBuf_`、继续复用 `rawBuf_`，预取会冲掉正在 Cast 的 half。

### 6.2 `count==1` 的 inf 不是慢路径

`unbias` 是标量，`n==1` 时为 `+inf`，随后 `Muls` 广播到 `batch_var`。`inf*0` 在 fp32 里是 NaN。reference 这段用 float64。核内是 fp32。两者是否在 ±inf/NaN 上与 910B 内置一致，只能对 bin，不能为了避免特殊值改乘法顺序或加 `eps` 分支。这条分支对性能没有贡献，误改会直接坏精度。

### 6.3 标量 V_S 环

每 tile 固定 `V_S` 与 `S_V` 各一次，再加 `len` 次标量 UB 访问。它把向量流水和 MTE3 隔开，使前后的 DMA 更难重叠。

- 小 `len`：环的绝对时间小，但两次排空会让「看起来能双缓冲」的改动测不出收益。这是小 shape 豁免的技术理由，不是借口。
- 大 `len`：`O(len)` 标量可能真占 A 类时间，才进入 K3。
- 实现风险：`GetValue/SetValue` 是当前唯一写明「量化之后补洞」的路径。Compare/Select 对 NaN 的输出位型在 910B 上要单独取证，不能从别的算子推断。

### 6.4 worldSize 串行累加

累加必须在量化之后的 fp32 上进行（先加再量化会改变 `Test_050` 一类的舍入）。所以设备循环里每一次都是「DMA + 量化 Cast + Add」，不能改成「先把 W 行 fp16 加完再量化一次」，除非另做 bit 级证明。那是语义实验，不是性能快路径。

串行本身在 `W=1` 时只有一圈，优化空间在拷贝与 Cast 的气泡（K1），不在减少圈数。`W>1` 才有 K2 的重叠。把 W 切到多核会引入 workspace 和第二轮归约，字节量通常大于核内多几次 `Add`。

`SumSampleCount` 在每核用标量 `GetValue` 扫 GM。`countNum` 为 1（样本）时可以忽略。它和世界循环不是同一个热点，不要绑在 K2 里一起改。

### 6.5 单 tile 与对齐尾

`calcCount = AlignUp(len, 8)` 参与向量运算，写出只用 `len`。尾车道必须保持 0，否则会污染 `Add` 进累加器再被截掉——截掉前若和 NaN 判定搅在一起会难查。性能改动若扩大 `calcCount` 或改 Pad，要带一个 `len` 非 8 倍数的点（例如 C=6）做正确性，但**不要**把该点的耗时写进性能表。

---

## 7. 废弃建议清单

无新证据（本题 profile 或任务书新 shape）不得重开。

| ID | 建议废弃 | 原因 |
|----|----------|------|
| D1 | 以 `samples` `[1,6]` 为目标的任何性能改动 | 正确性点；1 核、24B 级、`W=1`。不能代表门禁 |
| D2 | 未过 §1.1 四条件就做的双缓冲、事件重叠、向量补 NaN | 违反性能止损门禁。基线精度尚未结案 |
| D3 | 直接套用其他题的「≥95%」或「差 3us 即豁免」 | 那是 PRelu 示例条款。本题条文未入库 |
| D4 | 删除或跳过 fp16 量化的 fp32 快路径 | 破坏 `100.001→100.0`、`invstd=10000` |
| D5 | 将 `count==1` 无偏系数改成有限值 | 与 reference / 头注释 / 实测描述相反 |
| D6 | 在缺少 `W>1` 性能点时做跨设备双缓冲或按 world 切核 | `W=1` 无第二拍；切核的 GM 归约更贵。等任务书 shape |
| D7 | 单通道分核、撤销 `kMinBlock=8` | 与现有 tiling 的短向量判断相反，且样本会变成 6 个空核问题 |
| D8 | Cube、SIMT、通道维 atomic | 无对应算术结构 |
| D9 | 先加再量化，或把多设备 sum 合成一次 Cast | 舍入顺序是精度的一部分，未做 bit 证明前不是性能手段 |
| D10 | 把标量补 NaN 移到 `QuantFp16Vec` 或 fp16 store Cast 之前以「方便流水」 | Cast 会吞 NaN；fp16 `batch_invstd` 在补洞后还有一次 `CAST_RINT` |
| D11 | 向量化 `SumSampleCount` | 每核一次；「求和 / 取首元素」尚未钉死 |
| D12 | 跨 tile 双缓冲作为默认下一刀 | 每核单 tile 时重叠窗口为 0，还挤占 `maxTile` |

仍可在过闸且 profile 支持后**有限**重开的，只有 §4 的 K1～K5，并且每条单独记账、单独止损。K5 在 D6/D12 的证据出现之前保持废弃。

---

## 8. 本路结论（供汇总，不代替 SUMMARY）

1. 最简正确实现就是当前单缓冲 kernel。下一闸是云上冒烟和 reference/内置对拍，不是改流水线。
2. 性能工作在功能精度结案、同机 TBE 表、本题任务书门禁数字三者齐备之后才开始。开始时先 K0 测量。
3. 真正的结构特征是：DMA 与计算被立刻 Wait 钉死、`worldSize` 短循环、量化 Cast 绑在 `rawBuf_` 上、标量补 NaN 每 tile 排空两次。双缓冲只有在「多 tile 或 `W>1`」且 A 类未达标时才有重叠窗口。
4. `C=6` 走正确性，不走性能。小 shape 是否豁免，等本题任务书条款，不借用 PRelu 的 10us/3us。
