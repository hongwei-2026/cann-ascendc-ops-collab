# AI-1 分析 — SyncBatchNormGatherStats_2332（Composer · 正确性/精度铁律）

| 项 | 内容 |
|----|------|
| 题名 | SyncBatchNormGatherStats_2332 |
| 视角 | **正确性与精度铁律优先**；以 `samples/reference.py` 为权威 |
| 代码只读基线 | `people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/` |
| 日期 | 2026-11-17 |
| 隔离 | 本题专用；不引用他题优化数字或实测结论 |

---

## 1. 语义与验收门禁理解

### 1.1 算子语义（业务）

分布式 BatchNorm 统计汇总：在已知各设备（或单卡）上预聚合的 `total_sum`、`total_square_sum` 与全局 `sample_count` 下，计算当前 batch 的逐通道 `batch_mean`、`batch_invstd`，并按动量 `momentum`、稳定项 `eps` 更新 running 的 `new_mean`、`new_variance`。

公式（**中间用 fp32；符号在 fp16 量化后的输入上定义**）：

```text
sum_c  = Σ_devices total_sum[device, c]     （每设备元素先 fp16 量化再加）
sq_c   = Σ_devices total_square_sum[...]    （同上）
n      = sample_count（见 §1.4 争议点）
batch_mean   = sum_c / n
batch_var    = sq_c / n - batch_mean²
batch_invstd = 1 / sqrt(batch_var + eps)
new_mean     = (1-m) * mean_c + m * batch_mean
new_var      = n/(n-1) * m * batch_var + (1-m) * variance_c   （n==1 时 n/(n-1)=+inf）
```

四个输出在写出前再次 **fp16 舍入**；`batch_mean` / `batch_invstd` 的 **存储 dtype** 与 `total_sum` 等浮点输入一致；`new_mean` / `new_variance` 在 OpDef 中 **恒为 float32**，但数值仍在 fp16 网格上（fp32 用例输出位模式即「float 容器装 fp16 值」）。

### 1.2 权威参考与设备对齐

`samples/reference.py` 头部注释说明：与 **910B 内置算子**逐 bin 对齐（含 fp32 用例被图编译 Cast 进同一 fp16 语义核）。验收时应以该文件为 golden 生成器，而非「教科书 BatchNorm fp32 公式」。

关键反直觉用例（**Test_050 类**）：

- `total_square_sum` 经 fp16 量化后 `100.001 → 100.0`，使 `batch_var` **恰好为 0**；
- 此时 `batch_invstd = 1/sqrt(eps) ≈ 10000`（设备实测），**不是** fp32 全精度下的 ~315；
- 若实现跳过输入 fp16 量化，会在该类用例上 **确定性 FAIL**，也可能在部分 case 上因宽松 tol **假 PASS**（见 §5）。

### 1.3 OpDef dtype 与 shape（对照 `sync_batch_norm_gather_stats_def.cpp`）

| 输入/输出 | 支持的 dtype 组合（4 路 tiling） |
|-----------|----------------------------------|
| `total_sum`, `total_square_sum`, `mean`, `variance` | float 或 float16 |
| `sample_count` | int32 或 int64（与 float/float16 输入两两组合） |
| `batch_mean`, `batch_invstd` | 与浮点输入 **同 dtype** |
| `new_mean`, `new_variance` | **恒 float32**（4 组皆为 DT_FLOAT） |
| `momentum`, `eps` | optional float attr，默认 0.1 / 1e-5 |

`total_sum` / `total_square_sum` 逻辑形状：**末维为通道 C**，Leading 维乘积为 `world_size`（设备数）；可为 `[C]`、`[1,C]`、`[W,C]` 等。`InferShape` 使四路输出 shape **与 `mean` 一致**（逐通道，不保留 world 维）。

### 1.4 小 shape / 性能门禁（本题）

- `samples/case.json` 首点：**C=6，world_size=1，count=100** — 仅适合 **正确性冒烟**，`CLOUD-RUN.md` 已写明 **禁止用该 shape 做性能结论**。
- 性能专题需满足 `docs/collaboration/02-perf-gate.md`：功能+精度全绿、TBE 对照、台账「采纳（功能精度）」后才开始；小 shape（B 类）默认仿真+分析豁免，不无限抠 us。
- **本题当前阶段**：基线声称已按 fp16 语义实现；P0 应为 **云 `-e` + 全量 Judge/reference 对齐**，非改 tiling 追分。

### 1.5 reference.py 与 Kernel 的已知语义差（验收时必须显式测）

| 点 | reference.py | 基线 Kernel / Tiling | 风险 |
|----|--------------|----------------------|------|
| 多设备 `total_sum` | `reshape(-1)` 压平，**未**按通道跨 device 累加；注释写明用例恒 `[1,C]` | `worldSize` 循环 `offset = device * channels + channel` 量化后累加 | **world_size>1** 时 reference **不能**当 golden，必须以设备/bin 或扩写 reference |
| `sample_count` 多元素 | 仅 `reshape(-1)[0]` | `SumSampleCount()` **求和所有元素** | `[world_size]` 每卡 count 不等时，reference 与核 **可能分歧** |
| `n=0` | 未强调 | `invN = 1.0f / n` | 除零 / NaN；需 Judge 是否覆盖 |

---

## 2. 最简正确实现路径（基线评估）

### 2.1 基线是否「已够正确」（只读结论）

ZIP 基线 **已不是空壳**：Host（Def / InferShape / Tiling）+ 双模板 Kernel（`half` / `float`）+ `reference.py` 文档化语义。从 `sync_batch_norm_gather_stats.h` 可读证据：

1. **输入 fp16 量化**：`LoadQuantized` 对 `T=float` 先 `DataCopyPad` 再 `QuantFp16Vec`（fp32→fp16 RINT→fp32）；对 `T=half` 走 half→fp32 `CAST_NONE`（fp16 网格恒等）。
2. **中间 fp32 向量**：`batch_mean`/`batch_var`/`invstd`/EMA 在 LocalTensor<float> 上完成。
3. **跨 device 累加**：`worldSize` 维与 Tiling 推导的 `worldSize` 一致（`total_sum` leading dim 乘积）。
4. **count==1**：`unbias = PositiveInf()`，不人为改有限值。
5. **输出 fp16 网格**：四路 `QuantFp16Vec` 后再写 GM；`batch_*` 经 `StoreAsInputDtype`，`new_*` 经 `StoreFloat`（容器 float，值为量化后 fp32 位模式）。
6. **NaN / inf 补丁**：`batch_var + eps < 0` → `batch_invstd` 写 QuietNaN；`== 0` 且向量 `Div` 得 0 → 补 `+inf`（对应 `1/sqrt(0)`）；注释说明避免向量 `Sqrt`/`Cast` 吞 NaN。

**P0 最简路径（若用户授权编码）**：

```text
不重构算法 → 云 build.sh -e 冒烟 → 平台全量用例 vs reference/内置
→ 针对性补测：count==1、负 batch_var+eps、fp16/fp32 四 dtype、world_size>1、Test_050 类
→ 仅在 FAIL 点最小 diff 修 Kernel/Host（仍遵守 fp16 铁律）
```

若冒烟全绿，则 **无需**「另写一套正确 Kernel」；若 FAIL，优先怀疑 **量化次序、count 聚合、invstd 边界、new_var 的 fp64 中间项**，而非 tiling 分核。

### 2.2 与 reference 逐步对齐检查表（Implementer 用）

| 步骤 | reference | Kernel 对应 | 必测 |
|------|-----------|-------------|------|
| 输入量化 | `_q16` | `QuantFp16Vec` / half load | fp32 输入边界小数 |
| batch_mean | `float32(sum_i/n)` | `Muls(accSum, invN)` | 与 sum 量化后一致 |
| batch_var | `sq_all - mean_all²` | `Sub(accSq, Mul(accSum,accSum))` | Test_050 → var=0 |
| invstd | `1/sqrt(var+eps)` | `Adds`+`Sqrt`+`Div` + §NaN 补丁 | var=-eps, var=0 |
| new_mean | fp32 组合后 `_q16` | `Muls`/`Add` + `QuantFp16Vec` | fp32 输出=fp16 网格值 |
| new_var | `float64` 算 unbias 项再 `float32` | 标量 `unbias` float32 | count==1 → ±inf；与 reference ulp 差 |
| 输出 dtype | `astype(in_dtype)` 仅 batch_* | `StoreAsInputDtype` / `StoreFloat` | 判题机严格 dtype |

**轻微 ulp 风险**：reference 在 `new_var` 中对 `unbias * m * var_res` 使用 `float64` 再截断 `float32`；Kernel 全程 fp32 标量——仅在极端 magnitudes 可能差 1 ulp，应以 Judge 为准，**勿为追 fp32 全精度改语义**。

---

## 3. 性能候选（有序、有止损）— **勿先做**

> 门禁：见 §1.4；下列 **全部标 P1/P2**，在 path-ledger「采纳（功能精度）」之前 **禁止** 实施。

| 顺序 | 候选 | 预期收益 | 止损 |
|------|------|----------|------|
| P1-1 | 维持 Tiling 已有「小 C 少核 + blockFactor≥8」策略，用云 profile 验证是否仍瓶颈在启动/同步 | 中（小 C） | 0.5 人日无 >2% → 结案 |
| P1-2 | 调 `ubFactor` / `maxTile` 上限（8192）与 UB 预算公式 | 中（大 C） | 不改 buf 数量前提下试 |
| P2-1 | 减少 `QuantFp16Vec` 往返次数（合并 load/store 阶段） | 低～中 | **极易破坏精度**；须全量回归 |
| P2-2 | 双缓冲 device 维 `DataCopy` | 中 | 2h 无证据 → B 类豁免 |
| P2-3 | 标量 NaN/inf 环与 event 同步精简 | 低 | 不得改变边界 bin |

**勿用 samples C=6 论证上述任一方向。**

---

## 4. 明确「不要先做」的反模式

| 反模式 | 为何错 |
|--------|--------|
| fp32 直通：输入不量化、输出不 `_q16` | 违背 910B 内置语义；Test_050 类必挂 |
| 「修」`count==1` 把 `n/(n-1)` 改成 1 或 clamp | 与 reference/设备实测相悖 |
| 删除 invstd NaN/inf 标量补丁，仅信向量 `Div` | Cast 可能吞 NaN；负 `var+eps` 路径 FAIL |
| 用 PyTorch 默认 BatchNorm 或纯 fp64 公式当 golden | dtype/shape/量化链不一致 |
| 假设 `reference.reshape(-1)` 适用于 **world_size>1** | 假 PASS 或多 device FAIL |
| `sample_count` 只读首元素而核求和（或反之）不验就改 | 多元素 count 语义漂移 |
| 跳过 fp16 用例四路 dtype，只跑 case.json float32 | OpDef 半路径未覆盖 |
| 基线 FAIL 前先改 tiling 分核追性能 | 违反 perf-gate 阶段 |
| 仅本地 `rtol=1e-4`（case.json）而不对 fp16 ulp / 边界 nan | 宽松 tol 掩盖量化错误 |
| 把 `new_mean`/`new_variance` 写成 fp16 storage | 违 OpDef；判题 dtype FAIL |

---

## 5. 本题特有风险（假 PASS 重点）

### 5.1 fp16 量化链

- **全链位置**：输入 load →（可选 multi-device sum）→ 中间 fp32 算术 → **四输出** `QuantFp16Vec` → 写 GM。
- **假 PASS**：仅在「好算」的 sum/sq 上对齐，但漏 **平方和量化** 或 **输出二次舍入**，在 Test_050（var 精确为 0）或大数/小数组合上暴露。
- **fp32 模板**：必须在 **同一语义** 下量化；不可因 `T=float` 走「真 fp32 输出」。

### 5.2 `count==1` → `+inf`

- `new_variance` 含 `inf * batch_var`；再 fp16 量化可能得 `inf`/`nan`；**判题器对 nan/inf 有特殊处理**（reference 注释）— 实现成有限值可能在部分通道「看起来接近」实则 bin 错。
- **假 PASS**：只测 count=100；从未跑 count=1 单测。

### 5.3 dtype / 存储 vs 数值

- `new_*` **必须** `float32` ndarray；数值仍是 fp16 网格（如 `0.09997559` = `fp16(0.1)`）。
- `batch_*` fp16 路径：`StoreAsInputDtype` half 需 RINT；fp32 路径写 **已量化** 的 float 位模式。
- OpDef 四组 + int32/int64 count：tilingKey MODE_0/1 仅分 float16/float；count 宽度由 `countIsInt64` 分支 — **int64 count 应用实测** 而非假定与 int32 相同布局。

### 5.4 `world_size` 维与 offset

- Tiling：`channels = last dim`，`world_size = product(leading dims)`。
- Kernel：`offset = device * channels_ + channel` — 要求 `total_sum`/`total_square_sum` GM 布局与 **先 device 后 channel** 连续存储一致；rank=1 时 world_size=1。
- **假 PASS**：只用 `[1,C]`；多卡 shape 错 offset 时部分通道静默错。

### 5.5 `batch_invstd` 边界

- `batch_var + eps < 0`（量化后可能）→ **NaN**。
- `batch_var + eps == 0` → **+inf**（非 0）。
- 向量 `Sqrt` 对负数可能变 0 再 `Div` 得 0 — **无标量补丁会 FAIL**；有补丁时若误用 `batch_var` 未加 eps 判据会错。

### 5.6 其它

- `sampleCount_=0`：`SumSampleCount` 得 0 → `invN` Inf/NaN；需确认 Judge 是否合法输入。
- `momentum`/`eps` attr：Kernel 用 float bits 传递，未 fp16 量化 — 与 reference `float64` 读 attr 通常一致，非首要风险。
- **云端日志未归档即宣称 PASS**：违反仓规； teammates 无法复核。

---

## 6. 废弃建议清单（无新证据不得重开）

| ID | 方向 | 废弃理由 |
|----|------|----------|
| D-01 | 纯 fp32 语义无 fp16 量化 | 与 reference/910B 内置不一致 |
| D-02 | count==1 有界/置零无偏系数 | 设备明确要求 +inf 行为 |
| D-03 | 仅用 reference `reshape(-1)` 实现多 device 聚合 | 语义不完整；应用 per-channel 跨 device 和（Kernel 已对） |
| D-04 | 去掉 invstd 标量 NaN/inf 修补 | 向量+Cast 路径丢特殊值 |
| D-05 | 输出跳过终端 `QuantFp16Vec` | 四输出均需 fp16 网格 |
| D-06 | 未全量 dtype/count 边界即做 tiling/双缓冲性能 | perf-gate 阶段错误 |
| D-07 | 用 case.json 单点 + 宽松 tol 作为「50 点全过」依据 | 假 PASS |
| D-08 | 将 new_mean/new_variance OpDef 改为 float16 | 与 Def 冲突 |

---

## 7. 本路结论（供 SUMMARY 吸收）

1. **共识基线**：现有 Kernel 设计 **意图上已覆盖** P0 精度铁律（fp16 量化、fp32 中间、count==1、输出舍入、invstd 边界）；首要工作是 **验证** 而非重写。
2. **必补验证**：count==1、负 var+eps、四 dtype、world_size>1、`sample_count` 聚合语义、Test_050 类 fp16 平方和。
3. **reference 局限**：多 device / 多元素 count 不能盲信当前 reference 单行实现；应以设备 bin 或扩展 reference 对齐 Kernel。
4. **性能**：默认 **勿先做**；小 shape 不证性能。
5. **最大假 PASS 源**：跳过输入量化、漏 count==1/invstd 边界、只跑首条 float32 小 shape。

---

## 8. 引用文件（本题内）

| 文件 | 用途 |
|------|------|
| `samples/reference.py` | 权威语义与 Test_050 说明 |
| `op_host/sync_batch_norm_gather_stats_def.cpp` | dtype 四组合 |
| `op_kernel/sync_batch_norm_gather_stats.h` | QuantFp16Vec、累加、count==1、NaN 补丁 |
| `op_host/sync_batch_norm_gather_stats_tiling.cpp` | world_size / countNum |
| `samples/case.json` | 冒烟 shape；tol 0.0001 |