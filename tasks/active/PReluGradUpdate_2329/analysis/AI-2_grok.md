# AI-2 优化分析 — PReluGradUpdate

**路线一句话：** 这是带宽受限的逐元素 kernel。切分单位是「不跨通道的连续行片」；每片只有一个标量斜率，用一次逻辑 `>`、一次标量 `Muls`、两次 `Select` 在 UB 里用 fp32 算完，再写回。权重常驻或按片取标量，不展开到 GM。

模板里的扁平 `totalNum / blockFactor / ubFactor`、`SetBlockDim(1)`、bf16 走 `half`，都不能当起点。

语义（与 `samples/reference.py`、TBE `after_v200` 一致）：

```text
dx     = features > 0 ? grads : weights * grads
update = features > 0 ? 0     : features * grads
```

`features == 0`、`-0`、NaN 都走 else（NaN 的比较为假）。`update` 与 `features` 同形，通道维不求和。三输入同 dtype，OpDef 三条签名是 fp16 / fp32 / bf16 的并行组合，不是混精。FORMAT_ND，通道轴固定为 **dim 1**，广播形为 `[1, C, 1, ...]`。参考实现先升 fp32，再 cast 回声明 dtype。

---

## 1. 算法与数据流风险

**谓词必须是逻辑大于，且不能改写成 `< 0`。** `CompareScalar(..., CMPMODE::GT)` 作用在浮点值上。把存储位型转成整数再比，负数的符号位会让「负 > 0」成立，`-0`（fp16 的 `0x8000`）也会比 `+0` 大。这正是赛题要从二进制比较改成逻辑比较的原因。

`features < 0` 也不是 `!(features > 0)`：NaN 对两个关系都为假。用 `<` 会把 NaN 的 `update` 收成 0，参考实现给出的是 `NaN * grads`。`+0` / `-0` 都 `> 0` 为假，else 里照样做乘法，有符号零才和 numpy 一致。首条用例就固定四点：正、负、`+0`、NaN，并核对 `Select` 的位极性（Compare 真→bit 1→`src0`）。极性反了也能跑完，是静默错。

**TBE 的 0/1 张量公式只是数值规格，不是指令表。** `zero_one`、`one_zero`、加法再乘，比「一次掩码 + 两次 Select」多一轮比较和更多 UB。不要在 GM 上落地这些中间张量。

**流量是 4 倍主张量。** 每个元素读 `grads`、`features`，写 `dx`、`update`，权重只占 `C`。fp32 约 16 B/元素，fp16/bf16 存储约 8 B/元素。算术强度远低于 1 FLOP/B，屋顶在 HBM，不在向量 ALU，更不在 Cube。全核、对齐 DMA、双缓冲预取做对，就该落在 TBE 的同一屋顶上；95% 是调度问题，不是算法问题。

**行片一旦跨通道，标量斜率就是错的。** 为了 32 B 对齐去 `DataCopy` 进下一通道再整段 `Muls`，多出来的元素会吃到错误的 `w[c]`。对齐只允许发生在同一通道内部；行尾用 `DataCopyPad` + 精确 count，算完不写越界。

**InferShape 现在是生成器残留。** 循环把 `dx` 收成 `grads`、`update` 收成 `features`。合法输入二者同形，所以样例能蒙混；泛化时两边必须都等于 **features**。同时拒绝：`rank < 2`（参考实现直接抛错）、`weights` 非一维、`weights.length != dim1`、`grads.shape != features.shape`。空张量（任一维为 0）在 Host 上直接成功返回，Kernel 看到 `totalNum == 0` 立刻退出，禁止用 `inner` 做除数。

**偏移用 int64。** 单 tile 的 count 可以是 uint32（UB 片很小），GM 起点不行。大 `N*C*H*W` 用 uint32 偏移会回绕，表现为偶发错位，不是稳定的精度差。

**Workspace 保持 0。** 没有归约、没有跨核依赖，`SyncAll` 和 atomic 都是纯开销。

---

## 2. 推荐切分与多核

定义：

```text
C     = shape[1]                 // 必须等于 weights.length
inner = ∏ shape[2:]              // rank==2 时为 1
rows  = shape[0] * C             // 通道行数
一行  = 连续 inner 个元素，共享同一个 w[row % C]
```

内存序是行主序：行 `r` 的起点是 `r * inner`，通道号是 `r % C`。通道号在 **行号** 上取模一次，不在元素上做除法。

### 两种 tile，Host 二选一

| 模式 | 触发 | tile 内容 | 权重 |
|------|------|-----------|------|
| `ROW_SLICE` | `inner >= 128` | 一行内部的连续片，长度 `ubFactor`，最后一片更短 | 每片一个标量 |
| `PACK_ROWS` | `inner < 128`（含 rank-2 的 `inner == 1`） | 从行边界开始的若干**整行** | 片内按行重复 |

128 是冻结阈值：一行短于一条向量流水时，按行 `Duplicate` 的启动成本盖过计算。只允许用一条全核 profile 改一次这个常数，然后写死。

`ubFactor` 以元素计，向下对齐到 256 B（fp32 为 64 个数，fp16/bf16 存储搬运按 128 个数对齐；UB 内 fp32 主循环同样按 64 对齐）。`ROW_SLICE` 还要 `ubFactor = min(ubFactor, inner)` 的向下对齐值；若 `inner` 本身小于对齐长度，整行进 `PACK_ROWS`，不要为了对齐把两行拼进同一标量片。

### 核间：按「片」均分，不按 N

只切 batch 时，`N = 1` 的特征图（验收里最常见的全核大 shape）只会占 1 个核。扁平元素切分则会把片切到通道中间。

```text
chunksPerRow = ceil(inner / ubFactor)     // PACK_ROWS 时每片含 rowsPerTile 整行，chunks = ceil(rows / rowsPerTile)
usedCores    = min(GetCoreNumAiv(), chunks)   // 但见下方字节下限
blockFactor  = ceil(chunks / usedCores)
核 i         → 片 [i*blockFactor, min((i+1)*blockFactor, chunks))
```

片号还原：

```text
ROW_SLICE:  row = chunk / chunksPerRow
            off = (chunk % chunksPerRow) * ubFactor
            len = min(ubFactor, inner - off)
            w   = weight[row % C]
PACK_ROWS:  row0 = chunk * rowsPerTile
            连续 rowsPerTile * inner 个元素，起点必在行边界
```

核数以 `GetCoreNumAiv()` 为准（910B 不同 SKU 约 40～48），禁止写死 1 或 40。每核工作量低于约 64 KB 输入+输出时减少 `usedCores`：赛题要求的是「全核场景 ≥ 95%」，不是强迫 `[4,6,8]` 铺满所有核。过切会把小 shape 推进 `<10 us` 的固定开销里。

`SetBlockDim(usedCores)`。末核片数更少是正常的，每元素只被一个核写一次。

### TilingData 至少要有这些字段

现有三字段表达不了「行号 → 通道号」。建议：

```text
n, c, inner, rows, totalNum
ubFactor, rowsPerTile, chunksPerRow
coreNum, blockFactor
tileMode          // 0 ROW_SLICE, 1 PACK_ROWS
dtypeKey          // 与 tiling key 一致，仅作校验
```

改结构后同步 kernel UT fixture。`totalNum == 0` 时 `SetBlockDim(1)` 且 kernel 空返回。

---

## 3. Kernel 调度（UB、双缓冲、比较/选择/乘）

模板的 `BUFFER_NUM = 2` 只挂了一对 `VECIN/VECOUT`，两个输出会互相覆盖，深度 2 也不会自动重叠。

**队列（参与 GM 流水，深度 2）：** `gradsIn`、`featuresIn`、`dxOut`、`updateOut`。  
**不进队列：** 权重标量或 PACK 用的权重小缓冲；一块 Init 时 `Duplicate` 好的 fp32 零，长度 `ubFactor`；一条 Compare 掩码（uint8 bitmask，按 `AlignUp(count, 256) / 8` 预算，不要做成 fp32 的 0/1 张量）。

910B UB 以 `GetCoreMemSize(UB)` 为准，量级约 192 KB。按下面的活缓冲估 `ubFactor`，留约 10% 给队列头和权重：

- 四条双缓冲队列：fp32 路径约 `4 * 2 * 4 * ubFactor`
- 一条可复用的 fp32 临时（`w*grads` 与 `features*grads` 串行共用）
- 零缓冲一份、掩码一份
- fp16/bf16 还要加上 Cast 前后的存储宽度；保守按「队列用存储 dtype、计算用 fp32」算，fp16 的 `ubFactor` 先卡在 **4096**（约 150 KB 内），实测 UB 后再上调，不要一开始按 192 KB 顶满

**必须预取，深度 2 才有意义：**

```text
CopyIn(片0 的 grads 与 features)
for i in 本核的片:
    if i+1 存在: CopyIn(片 i+1)     // 两个输入一起入队
    Compute(i)                      // DeQue 本片两个输入
    CopyOut(i)                      // dx、update 都 EnQue 后再进入下一次 Compute
```

同一片的两个输入都到齐再算。只预取 grads、features 还在飞，向量流水会空转。

**单片计算顺序（掩码只算一次）：**

```text
1. 存储 dtype → fp32（fp32 输入跳过）
2. mask = CompareScalar(features, 0, GT)          // 主循环 count 对齐 64 个 fp32；行尾单独短 count
3. tmp  = Muls(grads, w)                          // ROW_SLICE：w 为标量
       或 Mul(grads, wTile)                       // PACK_ROWS
4. dx  = Select(mask, grads, tmp)                 // bit1 → grads，bit0 → w*grads
5. tmp  = Mul(features, grads)                    // 覆盖步骤 3 的 tmp
6. upd = Select(mask, zero, tmp)                  // bit1 → 0，bit0 → f*g
7. fp32 → 声明 dtype（fp32 输出跳过），CopyOut
```

步骤 4 之后 `grads` 仍要留给步骤 5，所以 `Muls` 不能原地写进 `grads`。`dx` / `update` 直接写入即将出队的输出缓冲。`update` 的零分支用常驻零张量做 tensor–tensor `Select`，第一版不做掩码按位取反：bitmask 的 NOT 容易按字节取反，和「1 bit 一元素」布局不符。

正分支的 `dx` 在 fp16/bf16 上经 fp32 往返仍是 bit-exact，可以和负分支一起 Cast，不必为了「正分支直接拷贝」拆两套 store。

---

## 4. 权重广播（Vector）

权重不写成 `[N,C,...]` 的 GM workspace。那会多一轮整张量写和读，流量翻倍，屋顶直接掉到 50% 以下，95% 门禁失败。

**`ROW_SLICE`：每片 `GetValue(row % C)` 一次，再 `Muls`。** 片长 ≥ 128，标量装载被摊平。大 `C` 也不占 UB。不要为这个路径把整个 `weights[C]` 抬进 UB。

**`PACK_ROWS`：**

- `inner == 1`（形状 `[N,C]`）：每一行与 `weights[C]` **等长对齐**，就是向量 `Mul`，没有空间广播。多行 tile 从行边界开始，UB 内把 `weights` 做倍增拷贝铺到 tile 长度（`C → 2C → 4C …`），每片一次，纯 UB 带宽。
- `1 < inner < 128`：对片内每个通道 `Duplicate(w[c], inner)` 填到行偏移。`inner` 小、`C` 大时这是标量段循环，只服务正确性；这种 shape 的字节数通常进不了全核门禁。不要为此上 Gather 索引表。

`GetValue` / `Duplicate` 的索引是行号模 `C`，在 Host 已经保证片不跨「错误的通道边界」（`ROW_SLICE` 不跨行，`PACK_ROWS` 只在行边界切）。

---

## 5. dtype / bf16

| 存储 | tiling key | Kernel 类型 | UB 数学 | 回写 |
|------|------------|-------------|---------|------|
| fp16 | MODE_0 | `half` | Cast → fp32 | Cast → fp16，RINT |
| fp32 | MODE_1 | `float` | 直接算 | 直接写 |
| bf16 | **MODE_2（新增）** | `bfloat16_t` | Cast → fp32 | Cast → bf16，RINT |

模板把 bf16 和 fp16 一起送进 MODE_0，实例化的是 `half`。bf16 的位型会被当成 fp16 解读，比较和乘法都是错的。`ASCENDC_TPL_UINT_DECL` 目前只有 0 和 1，不加 MODE_2 就没有 bf16 的二进制入口。

一个 `template <typename T>` 共享调度即可，入口类型必须是真实存储类型。比较放在 **升 fp32 之后**，这样 NaN 的传播与 numpy `astype(float32)` 再比较同一条路。bf16→fp32→bf16 对有限值和 NaN 载荷是精确往返；fp16 同理。会出舍入差的只有 else 分支的乘法，必须在 fp32 里乘完再 RINT cast。在 fp16/bf16 里乘，会在 65504 附近饱和，和「fp32 乘完再 cast」对不上。

权重在使用前升 fp32：`ROW_SLICE` 标量转换一次；`PACK_ROWS` 把本片用到的 `C` 段转换一次。

Cast 使用默认 round-to-nearest-even。验收若抽查 binade 边界，负分支 `w * grads` 是唯一敏感点；正分支 `dx = grads` 与正分支 `update = 0` 不产生舍入。

---

## 6. 性能路径与止损

先功能后性能。功能未全绿、没有同机 TBE 对照表时，不开性能专题。

**P0 基线就应该打过全核 95%。** 内容就是上文：行片切分、全核、四队列预取、fp32 数学、bf16 独立 key、workspace = 0。用大 shape 测（`inner` 为 32 B 整数倍，`rows * chunksPerRow` ≥ AIV 数，例如 `[1,16,128,128]` fp32、`[8,64,64,64]` fp16）。样例 `[4,6,8]` 只做正确性冒烟，它是 `<10 us` 档。

达标即停。继续抠指令不会在带宽屋顶上稳定超过 TBE。

**未达标时只查这四件事（合计 ≤ 1 人日）：**

1. `usedCores` 是否真等于 AIV 数，每核字节是否差一倍以上。单核耗时 × 核数若接近全核耗时，说明没有重叠或几乎没铺开。
2. trace 里 MTE2 / Vector / MTE3 是否错开。三阶段串行就是预取没写上，与 `BUFFER_NUM` 无关。
3. 门禁 shape 的 GM 起点是否 32 B 对齐。`inner * sizeof(T)` 能被 32 整除时，行首自然对齐；`ubFactor` 保持 256 B 对齐。
4. `ubFactor` 是否小到每次启动都在付流水税。在 UB 占用约 70%～80% 的档位停，不顶满。

这四项相对上一版增益 `< 2%` 就停，写 DEC：瓶颈在带宽且已与 TBE 同屋顶，或指出具体未对齐的那个验收 shape。

**结构性改动只保留一个候选，时限 ≤ 2 人日，且必须先有 trace 证明向量侧堵、MTE 空闲：** 把 `Select` 融合进更少的向量指令。没有这份 trace 就不做。做完增益 `< 2%` 同样回退到 P0 二进制。

**`<10 us` 且与 TBE 相差 ≤ 3 us：** 2 小时内取仿真/profiling，写明 launch+tiling 固定开销、核函数字节量 `4 * numel * sizeof(T)`、与 TBE 同一流量。然后停止。禁止为小 shape 加第二套微核。

诊断口径：Ascend C 带宽达到同机 TBE 的约 70% 以上、核数已经铺满、流水已经重叠，就认为门禁可达，停止找新算法。

---

## 7. 不要先做的事

1. 按 TBE Python 落地 `zero_one` / `one_zero` 两条掩码再相加。
2. 把权重 `Broadcast` 成与 `features` 同形的 workspace。
3. 在核内对每个元素做 `/ inner % C`。
4. 维持 `SetBlockDim(1)`，或只沿 `N` 切分。
5. bf16 复用 `half` kernel；用整数比较代替 `CMPMODE::GT`。
6. 在 fp16/bf16 里做乘法来省 Cast。
7. 对 `update` 做通道归约、atomic 或 `SyncAll`（旧 aclnn「update 为 `[C]`」语义，内置算子不接受）。
8. 把 `[C]` 右对齐到最后一维。用 `[2,3,5,7]`、`weights.length == 3` 做反例。
9. 只调通样例 `[4,6,8]`，不覆盖 rank-2、`C = 1`、`N = 1`、`inner` 非对齐、空张量、bf16。
10. 为对齐 DMA 让 `ROW_SLICE` 读进下一通道。
11. 上 Cube、L1、NZ。本算子没有可复用的矩阵块。
12. 在基线对照表出来之前改指令融合、改 128 这个阈值、或为 `<10 us` 开专题。

---

## 动作清单

**P0（功能基线，同时就是性能基线）**

1. InferShape：`dx`、`update` 都等于 features；rank、`C`、同形检查；空张量短路。
2. 扩展 TilingData：`n/c/inner/rows/tileMode/ubFactor/chunksPerRow/blockFactor`；`GetCoreNumAiv()`；小字节降核；workspace = 0。
3. Kernel：四队列双缓冲预取；`ROW_SLICE` 用标量 `Muls`，`PACK_ROWS` 用行对齐权重铺片；一次 GT 掩码 + 两次 Select；偏移 int64。
4. MODE_2 + `bfloat16_t`；三种 dtype 都在 fp32 中计算，fp16/bf16 RINT 写回。
5. 正确性：正/负/`+0`/`-0`/NaN 极性；通道在 dim 1 不在最后一维；行尾非对齐；rank-2；`N=1` 大 `inner` 的多核切分；三 dtype 各一条舍入（负分支乘法）。

**P1（仅当全核对照 < 95%，≤ 1 人日）**

6. 查核数、预取重叠、32 B 对齐、`ubFactor` 档位。增益 `< 2%` 写 DEC 并停止。

**P2（默认不做）**

7. 有向量侧 trace 才考虑压缩 Select；否则放弃。
8. `<10 us` 且差距 ≤ 3 us：仿真图 + 流量说明，2 小时封口。
