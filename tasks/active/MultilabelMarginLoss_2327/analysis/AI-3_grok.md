# AI-3（Grok）独立分析 — MultilabelMarginLoss problem 2327

视角：**多核归约安全、大 C 切分、TBE / `reference.py` 精度边界**。  
材料仅来自本题 `materials/_unzipped/code/` 与协作规范 `02-perf-gate.md` / `04-multi-ai-analysis.md`。  
**未读**其他 AI 分析；**不抄**其他题目门禁数字当本题事实。

---

## 1. 一句话路线

**先按行做「截断→计数→float32 加权 hinge」，行内用 pos 常驻 + neg 分块消化大 C；跨核只归约「已除以 C 的 per-row loss」，禁止把 target 前缀或同一 `y[i]` 拆给多核写。**

补充（相对「按行分核 + workspace Finalize」的默认叙事）：  
Finalize 只解决 **mean/sum 的核间加法**；本题更危险的切分点在 **K 维截断** 与 **C 维 pairwise**。行切分在 `N < 核数` 或 `C` 撑满 UB 时不够用，必须另开「行内 pair-tile」而不是把一行的 `target` 或 `y[i]` 均分给多核。

---

## 2. 语义（金标准：`samples/reference.py`，对齐 910B TBE）

模板注释写明：与部分 aclnn / PyTorch 文档有两处实质差异。冲突时以 `reference.py` 为准，不以文档「0/1 标记」为准。

### 2.1 形状与 dtype（OpDef + 样例）

| 张量 | 形状 | dtype | 备注 |
|------|------|--------|------|
| `x` | `[N, C]` | fp32 / fp16 / bf16 | `AutoContiguous`，ND |
| `target` | `[N, K]` | int32 | `K` 常等于 `C`，但语义按「可与 x 同形」写，**不保证 K==C** |
| `y` | `none→[N]`；`mean/sum→[1]` | 与 `x` 相同 | 样例 `case.json` / example：`reduction=mean`，`y` 为 `[1]` |
| `is_target` | **`[N, C]`**（跟 `x` 的 C，不是 K） | **恒 int32** | 与 `x` dtype 无关 |

`reduction` 可选字符串，默认 `"mean"`；数值约定 `0=none / 1=mean / 2=sum`。

InferShape 模板把两个输出都拷成对应输入形：`y←x` 变成 `[N,C]`，`is_target←target` 在 `K≠C` 时也会错。这是 Host 第一关，不是 Kernel 能补的。

### 2.2 计数型 `is_target` + 截断

对每一行 `i`：

1. 在 `target[i, 0:K)` **从左到右**找第一个 **`<0`** 的位置（不限于 `-1`），该位置及之后全部丢弃。无负数则扫描到 `K`。
2. 前缀里取值 `t` 且 **`0 ≤ t < C`** 的元素：`is_target[i, t] += 1`。
3. `t ≥ C` 或前缀外的合法类：**不计**。
4. 结果是 **出现次数**，不是 `{0,1}`。重复标签累加（注释：`[3,2,0,3,0,0] → [3,0,1,2,0,0]`；四个 `3` 计 4）。

`is_target` 判题：**整数 0 容差**（`case.json` / `sample.py`：`rtol=atol=tol=0`）。

### 2.3 加权 hinge

中间量 **一律 float32**（`x` 先升 fp32），再按输出 dtype 舍入。

```text
losses[i] = (1/C) * Σ_{j: cnt[j]==0} Σ_{l: cnt[l]>0} cnt[l] * max(0, 1 + x[j] - x[l])
```

- 分母是 **`C`（类数）**，不是 `K`、不是 `nPos`、不是有效前缀长。
- `cnt[l]` 把重复标签变成对同一 `x[l]` 的整数权重；与「按前缀位置展开再求和」等价。
- `cnt` 全 0（该行无有效正类）：`losses[i] = 0`，**仍占 mean 的分母 N**。
- 每个类恰好属于 `cnt==0` 或 `cnt>0` 之一，故 `nPos + nNeg = C`。复杂度最坏 `O(N·nNeg·nPos) ≤ O(N·C²/4)`。

`relu` 使双线性分解失败：不能先对正类做 `Σ cnt[l] x[l]` 再一次向量减。必须保留 **(负类 tile × 正类 tile)** 的两重循环（可向量化）。

### 2.4 reduction 输出形

- `none`：`y = losses`，形 `[N]`。
- `mean`：`y = [mean(losses)]`，形 **`[1]`**（`np.atleast_1d`；0 维标量会被判题机拒）。
- `sum`：`y = [sum(losses)]`，形 `[1]`。
- 先按行 `/C` 得到 `losses[i]`，再对 N 做 mean/sum。`reference.py` 行内 `sum` 用 **float64**，TBE 注释称内核 **float32**——这是精度边界，见 §6.5。

样例门限：`y` 的 `rtol=atol=1e-4`；`sample.py` 对浮点比较开了 **`equal_nan=True`**。正式赛题阈值「待 TASK 与平台核对」，分析阶段不当成已公布的别题数字。

---

## 3. 最简正确路径（P0）

目标：功能 + 与 `reference.py` 对拍，**单核或「一行一核且 N 很小」即可**，不追求占满 AIV。

1. **InferShape**  
   - 读 `x:[N,C]`、`target:[N,K]`、`reduction`。  
   - `is_target → [N,C]`。  
   - `y → [1]`（mean/sum）或 `[N]`（none）。  
   - 拒绝非法 reduction / 维数不是 2 / `N` 不一致。

2. **Tiling 最小集**  
   不要沿用模板把 `totalNum` 当「扁平元素数」切 1024（`MIN_SPLIT_THRESHOLD` 对 pairwise 无意义）。至少下发：`N, C, K, reduction, usedCores=1, ubTileC`。  
   **P0 workspace 可为 0**（单核直接写 `y`）。

3. **Kernel（三 dtype 共用一条 float32 计算链）**  
   对 `i in [0, N)`：  
   - 标量/小块扫 `target` 得 `end`；前缀 `bincount` 写整行 `is_target`（先清零）。  
   - 抽 `pos`（`cnt>0` 的 `x[l], cnt[l]`）常驻 UB（或 GM 再分块，见 §4）。  
   - `neg` 按 `ubTileC` 滑窗：`A = max(0, 1 + xj - xl)`，`acc += A * cnt[l]`（**fp32**）。  
   - `losses[i] = acc / C`（fp32 除法，再按输出 dtype 写）。  
   - `none`：写 `y[i]`（**标量/Pad 写**，禁止 32B `DataCopy` 踩邻行）。  
   - `mean/sum`：单核累加后再 `/N` 或直接 `sum`，写 `y[0]`。

4. **补齐 BF16**  
   OpDef 三键是 fp32 / fp16 / bf16。现网入口只实例化 `half` 与 `float`，且 tiling 把 **fp16 与 bf16 都映射到 schMode 0（half）**——P0 必须拆开，bf16 升 fp32 算、再截回 bf16。

5. **对拍用例（实现阶段，非本文件任务）**  
   小 `N,C`：截断、重复计数、`t≥C` 丢弃、空正类、`K≠C`、none/mean/sum、三 dtype、NaN/Inf。  
   大 `C` 至少一条「一行装不下全量 pairwise 中间矩阵」的分块路径。

---

## 4. 性能候选 + 止损

性能门禁在 TASK 标 **待核对**（同系列口述「全核 ≥ TBE 95%、小 shape 有豁免叙事」）。**在未拿到本题正式表之前，不把任何百分比/微秒写成已生效事实。**  
允许开性能专题的前提见 `docs/collaboration/02-perf-gate.md`：功能全绿、精度过阈值、**同机 TBE 对照表**、基线已在 path-ledger 结案。否则退回正确性。

### 4.1 分流（用本题结构分类，不填别题数字）

| 类 | 本题判定 | 策略 |
|----|----------|------|
| A | 正式表列出的全核 / 大 `N·C²` 点 | 必须对 TBE；才允许 P2 |
| B | 样例级 `[4,6]`、耗时极短、核空转 | 仿真图 + 分析即可，**禁止为占核去切 C** |
| C | 未进验收表的极端 `C`/`K` | 记录正确性，默认不抠 |

### 4.2 有序候选（每条先记账再做）

| 序 | 假设 | 做法 | 时限（规范默认） | 止损 |
|----|------|------|------------------|------|
| P1-a | 瓶颈在行并行度 | `N` 按核切；每核私有行的 `is_target` 与 `losses[i]` | 0.5～1 人日 | 无同机对照或增益 <2% 停 |
| P1-b | mean/sum 需要核间加 | **workspace 存 per-row 或 per-core 的 fp32 部分和**，**一核按行序 Finalize** | 同上 | 禁止改用无序 AtomicAdd 当「更快的 Finalize」 |
| P1-c | 行内 `C` 大、UB 紧 | pos 压缩常驻 + neg 滑窗 + 双缓冲搬运 `x`/`target` | 同上 | 不准为省搬数改用 fp16 算 hinge |
| P2-a | `N << 核数` 且 A 类未达标 | **只切 pair 空间**（`j` 块 × `l` 块），部分和进 workspace，**仍只允许一个核写 `y[i]`** | ≤2 人日且要 DEC | 连续两条互斥路失败 → DEC |
| P2-b | 归约顺序导致 1e-4 擦边 | 对齐 TBE 累加顺序（类下标递增 / 先 l 后 j），而不是换算法 | 短 profile | 无流水线证据停 |

### 4.3 止损触发（命中即停，24h 内 DEC）

- 时限到且相对上一版增益 < 2% 或无增益。  
- 两条互斥结构（例如「只切 N」vs「切 C 占核」）都未达 A 类。  
- 无 profile，只靠感觉改。  
- 为 B 类小 shape 已超过约 2 小时。  
- **未写 DEC 不得再开新性能道路。**

B 类默认走仿真豁免材料，不无限抠微秒。

---

## 5. 反模式（不要先做）

1. 把 `is_target` 做成 0/1，或抄 PyTorch `MultiLabelMarginLoss` 文档当 golden。  
2. 用第一个 `-1` 截断、忽略其它 `<0`；或截断后仍扫后面的合法标签。  
3. 分母用 `K` / `nPos` / 有效前缀长。  
4. InferShape：`y=[N,C]` 或 `is_target=[N,K]`。  
5. **把一行的 `target[0:K]` 按核切开去并行计数**（见 §6.1）。  
6. **多核直接 `AtomicAdd` 到 `y[0]`** 当 P0/P1（顺序依赖 + 与 TBE 累加链不一致）。  
7. 在 UB 展开整张 `C×C` hinge 矩阵。  
8. fp16/bf16 路径上用半精度算 `1+x[j]-x[l]`。  
9. 把 bf16 丢进 `half` 模板。  
10. `none` 下用整 block `DataCopy` 写长度为 1 的 `y[i]`。  
11. 把 `mean` 融合成一次 `ΣS_i/(N·C)`，未与 TBE 对拍就当等价。  
12. 跳过 SUMMARY / 基线，直接为占满核做 C 切分。  
13. 把其他题目的 95% / 时钟 / Select 结论写进本题设计当事实。  
14. 整段粘历史 ops-nn 实现当「已验收语义」（TASK 已禁止）。

---

## 6. 特有风险

### 6.1 截断是前缀，不是可交换的 scatter（比行切分更先的安全约束）

`end = first index with target<0`。若核 0 扫 `target[0:K/2)`、核 1 扫后半：

- 前半已出现 `-1` 时，后半的合法 `t` **必须全部作废**；后核不知道 `end` 仍会 `+=`。  
- 负数只在后半时，前核会把本应截断前的计数做完，看起来「碰巧对」，对测例不稳定。

**废弃：按 K 分核计数。**  
正确并行粒度：

1. **整行串行**（或单核）求出 `end`；  
2. 再对前缀计数。  
若前缀仍要多核，必须先广播 `end`，且只处理 `< end` 的下标——P0/P1 **不要做**。

`t≥C` 的过滤发生在截断**之后**；与 `<0` 截断是两道门，不能合成「非法值跳过并继续」。

### 6.2 mean/sum 跨核

`y` 只有一个标量槽。多核同时写 / 无序 AtomicAdd：

- 浮点加法不可交换，与 TBE 的「先形成 `losses[0..N)` 再 mean/sum」不一致；  
- `mean` 的分母是 **全局 N**，不是「本核行数」；空正类行损失为 0 **仍要除进去**；  
- 模板 `GetWorkspaceSize = 0`，多核归约会直接踩系统 workspace 或写坏。

推荐（仍属 Finalize，但存储单元要选对）：

- **优先 workspace 长度 `N` 的 fp32 `losses[]`**，一核按 `i=0..N-1` 再 reduce。这样行内累加顺序与单核基线一致，核间只做一次有序加。  
- per-core 标量和更省空间，但 **打乱行序**，只适合已证明 1e-4 很松的 A 类点。  
- SoftSync 后单核读回也可以，仍要保证 **只有一个核写 `y[0]`**，且用 Pad/标量写 4B。

`none` **禁止**走「大家写完再 Finalize `y`」的标量路径——`y[i]` 必须行主核独占。

### 6.3 `none` 多核写 `y[i]`

- 一行一核：下标不交才安全。  
- 一行多核（pair-tile）：各核只有 **部分和**，必须 workspace 归约后 **单一 owner** 写 `y[i]`。两个核各写「自己的部分 /C」会丢加数或互相覆盖。  
- `y` 在 none 下是 `N` 个 4 字节元素，**几乎一定不对齐 32B**。错误 `DataCopy` 会改写邻居 `y[i±1]`，表现为偶发、随 `N` 变化的 WA。  
- `is_target` 同行 `C` 个 int32 同样有尾块对齐问题，但至少是稠密行写；`y` 更窄、更险。

### 6.4 workspace

现状：Host 写死 `WS_SYS_SIZE=0`；Kernel `Init` **不接收 workspace**；example 按 aclnn 回报来 malloc。  
一旦出现：核间 reduce、`N=1` 的 pair 切分、`cnt`/`pos` 溢出 UB 的 GM 暂存——必须：

- Host 计算 **用户 workspace + 系统预留**；  
- 结构体带偏移（`wsLossN`、`wsPartial`、`wsPosList`）；  
- Kernel UT 的 32B 夹具同步加大，避免 CPU 仿真越界装成「偶发通过」。

Workspace 里的部分和必须是 **fp32**，不要按输出 dtype 存半精度再加。

### 6.5 TBE / reference 精度边界（本题核心）

| 层级 | 行为 | 对实现的含义 |
|------|------|----------------|
| TBE（注释） | 中间 **float32**，最后按输出 dtype 舍入 | Kernel 主累加器用 fp32 |
| `reference.py` | `A` 与权重在 fp32，**`.sum(dtype=np.float64)/C`**，mean/sum 再 cast | CPU golden 可能 **比 TBE 更准** |
| 样例 Judge | `y`：`isclose(..., rtol=1e-4, atol=1e-4, equal_nan=True)` | 宽松，但大 `C²` 对仍可能擦边 |
| `is_target` | 精确相等 | 与浮点策略无关 |

边界结论：

1. **实现应对齐 TBE 的 fp32 链**，不要为了贴 numpy 去用 fp64（设备端也不该作为 P0）。  
2. 对拍脚本若只用 `reference.py`，大 `C` 上可能出现「NPU≈TBE 但 vs numpy 1e-4 边缘」。需要 **同机 TBE 输出** 当性能/精度仲裁，而不是改算法去贴 float64。  
3. **加法顺序**：按类下标 `l` 升序、`j` 升序的 tile 与乱序 pair 在 fp32 下会差 ulp。P0 固定「`l` 外/`j` 内或与 TBE 向量化一致」并写进设计，避免性能阶段乱改顺序。  
4. **不要合并** `(Σ_i Σpairs)/ (N·C)`：多一次 `/C` 再 `/N` 与一次除，舍入不同。  
5. **权重 `cnt[l]` 先转 fp32 再乘**，避免 int32 累加后再转。  
6. fp16/bf16：**先升、后算、最后一轮 cast**。`1 + xj - xl` 在 fp16 对 `|x|` 稍大即 Inf，再 `max` 污染整行。  
7. 模板 tilingKey 把 bf16 当 half：是 **静默错键**，不是性能优化。

`reference.py` 声称与 910B「逐位一致」且用 numpy64 求和——以注释里的 **实测小用例** 为限；大 `C` 不要假设 bit-exact(numpy64, TBE)。

### 6.6 NaN / Inf

- `1 + xj - xl`：`Inf-Inf`、`Inf+(-Inf)` → NaN；`max(0, NaN)=NaN`，行损失变 NaN，mean 变 NaN。  
- 样例允许 `equal_nan`，故 **不要**把 NaN 洗成 0。  
- `x` 含 Inf：hinge 可持续为 Inf，`/C` 仍 Inf；cast 到 fp16 变 Inf。保持 IEEE，勿 clamp。  
- `is_target` 与浮点无关，NaN 输入不能改计数。  
- 除数 `C`：来自 `x.shape[1]`；`C=0` 属非法形，Host 拦截，Kernel 不要偷偷改成 `/nPos`。

### 6.7 dtype 三键

```text
schMode 必须三分：fp32 / fp16 / bf16
计算：CastTo<float> → hinge/acc/div → CastTo<T>
is_target：始终 int32 GlobalTensor，与 T 无关
target：始终 int32
```

现网：`MODE_0 = half`，`MODE_1 = float`，bf16 误入 half。UT 只跑 key=1、`[4,6]` fp32。P0 就要把三键与 Host `SetTilingKey` 对齐，否则「fp32 绿、提交三键红」。

### 6.8 `C` 过大装不下 UB

设一行工作集粗估（双缓冲时再 ×2）：

```text
x_row:    C * sizeof(T)     （计算前升 fp32 则 C*4）
cnt:      C * 4
target:   K * 4
pos 压缩: nPos * (4+4)      （xl + cnt）
neg tile: Tneg * 4
acc / 临时向量: 若干 * 32B 对齐
```

910B UB 量级在 Host 模板里按平台查询（UT 夹具给了 `262144` 仅作测试桩，**不是**可写进设计的硬件承诺）。  
`C=4096` 量级：一行 `x+cnt+target` 约数十 KB，**整行常驻往往可行**，装不下的是 **`C×C` 的 A 矩阵**。  
再大：

- **禁止**整行 `x`+`cnt` 强行 UB；  
- `cnt`/`x` 按类下标分块：先扫 target 把 `cnt` 写到 **GM `is_target`**（本就要输出），之后当 GM 计数表回读；  
- pos 列表若 `nPos` 仍过大，对 `l` 再分块，与 `j` 块做笛卡尔，部分和 fp32 留在 UB 或 workspace；  
- **切 C 分给多核**仅当 `N` 填不满核且 A 类需要：每核绑定 `(j-range, l-range)`，**不绑定**残缺的 target 前缀。

`nPos≈C`（几乎每类都出现）时压缩无收益，退化为两路滑窗，复杂度仍 `O(C²)`——这是算力墙，不是再切核能消失的。

### 6.9 其它 Host / 入口坑

- 模板 Kernel 未挂 `target` / `is_target` / `workspace` 的 GM。  
- `blockDim` 写死 1；`tiling` 三字段全是向量模板遗物。  
- aclnn example 只覆盖 mean + `[4,6]` fp32，**不能**当三键/三种 reduction 的充分集。  
- `is_target` 清零责任：多核若「各自 zero 全行再加」会互踩；应 **行主核 zero+count**，或 Host 保证 GM 初值并单核写。

---

## 7. P0 → P1 → P2

| 阶段 | 做 | 不做 |
|------|----|------|
| **P0 正确性** | InferShape；单核或「一行一核且不跨核 reduce」；截断+计数+加权 hinge；`/C` 后再 reduction；三 dtype 升 fp32；`y`/`is_target` 窄写安全；`K≠C`、空正类、重复标签 | 占满核；AtomicAdd；K 分核；C×C 矩阵；性能数字 |
| **P1 标准优化** | `N` 分核；`losses[N]` workspace + **有序** Finalize；pos 常驻 + neg tile + 双缓冲；按平台 UB 算 `ubTileC` | 为 `[4,6]` 抠 us；融合 `1/(N·C)`；bf16 当 half |
| **P2 仅 A 类未达标** | pair 空间切核（单一 `y[i]` owner）；对齐 TBE 累加序；DEC 后才动结构大改 | 无对照表开 P2；B 类死磕；重开 §8 废弃项 |

阶段门：无 SUMMARY、无基线结案 → 不准进 P2。

---

## 8. 废弃建议清单（无新证据不得重开）

1. 文档/PyTorch 版 **0/1 `is_target`**。  
2. **按 K 分核**扫描 `target`（截断不可交换）。  
3. **多核 AtomicAdd / 无序加到 `y[0]`** 作为功能基线。  
4. UB 内 **整幅 `C×C` hinge**。  
5. **半精度 hinge**（含「先 fp16 再累加到 fp32」）。  
6. **bf16 复用 half kernel**。  
7. InferShape **`y` 克隆 `x`、`is_target` 克隆 `target`**。  
8. 分母 **`K` / `nPos` / 前缀长度**。  
9. 未对拍就把 **`Σ/(N·C)`** 当 mean。  
10. 用 **别题性能数字** 或历史 ops-nn 回退实现当本题门禁/语义。  
11. 为 B 类小 shape **切 C 占核**。  
12. 多核同时 **整表 zero `is_target`**。

---

## 9. 材料事实摘要（便于汇总核对）

- 模板 Kernel / Tiling / InferShape 均为 TODO；功能路径需从零按 `reference.py` 写。  
- 首测点：`x,target [4,6]`，`reduction=mean`，`y[1] fp32`，`is_target[4,6] int32`，`y` 1e-4 / `is_target` 0。  
- 硬件配置键：`ascend910b`。合入预期 `cann/ops-nn`（以正式赛题为准）。  
- 本分析 **不修改** `design.md` / `SUMMARY.md` / 任何代码。
