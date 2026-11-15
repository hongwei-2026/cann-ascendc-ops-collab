# AI 交接：QuantUpdateScatter_2330

> 给后续 AI / 队友：背景、坑、已做、下一步。读完再改代码。  
> 座位：**只写** `people/yuhongming-2026/QuantUpdateScatter_2330/`；台账写本题 `tasks/active/QuantUpdateScatter_2330/`。

---

## 1. 题是什么

算子：`QuantUpdateScatter`（aclnn 侧常叫 `aclnnInplaceQuantScatter`）。

对 `var`(int8) 做「先拷贝，再按 indices 把 updates 量化写入」：

```text
y = copy(var)                          # 同址 inplace 可跳过
for each batch p, index idx:
  for k in update_axis:                # attr axis 仅支持 -2
    y[p, ..., idx+k, :] = int8_rint( updates[p,...,k,:] / scales + zp )
```

- `quant_axis` 仅 `-1`（最后一维）。
- 除法在 **fp32**；取整 **CAST_RINT**；再 clip `[-128,127]`。
- **不要**改成 `fp32→half→int8` 直接量化（二次舍入，与内置不一致）。内置路径是 `fp32→int16(RINT)→half→int8`。
- `reciprocal_scale=true`：用乘法代替除法。
- 两套 dtype：见 `op_kernel/quant_update_scatter.cpp` MODE_0 / MODE_1。

权威语义：`workdir/code/samples/reference.py` 文件头注释 + 探针对拍。

---

## 2. 仓库位置（勿串题）

| 用途 | 路径 |
|------|------|
| 代码 | `people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/` |
| 台账 | `tasks/active/QuantUpdateScatter_2330/`（STATUS / path-ledger / dev-log） |
| 云测命令 | 本题 `CLOUD-RUN.md` |
| 来源 ZIP | 本机 `D:\华为算子比赛\QuantUpdateScatter_2330.zip`（**不入库**） |

协作仓：`https://github.com/hongwei-2026/cann-ascendc-ops-collab`  
云端常见根：`/mnt/workspace/cann-ascendc-ops-collab`

---

## 3. 基线从哪来

- ZIP 内已是**可编译**功能实现（含 build 产物；入库时已剔除 `build/`）。
- UT 编译路径曾指向 `people/yuhongming-2026/...`，故座位定为 yuhongming。
- samples 首 case 极小：`var[1,2,32]` / `updates[1,1,32]`——**只验证正确性**，不能当性能结论。

---

## 4. ZIP 朴素版的性能问题（已诊断）

1. **`CopyVarToOut` 只跑 block0**：大 `var` 时整表串行拷贝 + `SyncAll`，极易成主耗时。  
2. **每行都重新 DMA scales/zp**：scales 只依赖 lastDim 下标，与行无关 → 重复搬运。  
3. **`indicesGm_.GetValue` 标量读**：batch 大时 GM 延迟明显。  
4. **MTE2 / V / MTE3 全屏障串行**：无双缓冲（留给 Q002）。  
5. **tiling 核数只跟 batchSize**：batch=1 但 var 很大时无法多核拷贝。

---

## 5. 已落地：Q001（待云测，勿谎称 PASS）

文件：

- `op_kernel/quant_update_scatter.h` — 主逻辑  
- `op_host/quant_update_scatter_tiling.cpp` — 核数 / UB 预留  
- `samples/reference.py` — 逐步 fp32 + reciprocal 直接乘（对齐 kernel）  
- 本地证据：`explorations/EXP_Q001-local.md`（CPU 回归；**非** NPU PASS）

改动要点（Cursor 初版 + Codex 本地加固后同步入主仓）：

| 项 | 做法 |
|----|------|
| 多核拷贝 | 按 **32B block** 分核，避免边界重叠；inplace 同址跳过 |
| tiling 抬核 | `usedCores = max(batchCores, copyCores)`，约 32KiB/核；**不要**再按 batch 缩回 1 |
| indices | 每核缓存 cap 2048；超限整段回退 GetValue |
| scales/zp | 外层按 lastDim tile 只 Load 一次，内层扫所有待写行复用 |
| 精度路径 | 仍保持 fp32→int16(RINT)→half→int8；`SyncAll` 保留 |
| UB | tile/缓冲按 32B 对齐；indices 最坏预留 16KiB |

**云测观察点**：尾块 `len` 未 8 对齐时 Vector API 是否接受（Codex 用真实 `len`，未强制 `AlignUp(8)`）。

**未做（有意留给后续 AI）**：双缓冲、多行合并 DMA、特化「无 zp / reciprocal」模板、与 TBE 同机对照。

### R001（平台测点2）

截图：测点1 Pass、测点2 **Runtime Error / ACL 161002**（参数非法）。  
修复：`op_host/quant_update_scatter_def.cpp` 注册 **4** 个配对签名（fp16/bf16 × int32/int64）；`op_kernel/quant_update_scatter.cpp` 按 `tilingData.indicesIsInt64` 选模板。详见 `explorations/EXP_R001-runtime.md`。  
平台提交 **1345715** 已 Pass（50 点）。

### Q002（冲时延）

- 可见点仍慢于榜最优；候选：`TQue` 两槽 + `workBuf` 作整表拷贝 scratch（更大 DMA 块）。
- **非**真·跨行双缓冲（每 tile 仍 EnQue 后立刻 DeQue）；主要收益待测。
- `TQue` 模板深度须与 `InitBuffer(..., 2)` 一致（已改为 2）。
- 止损：见 `decisions/DEC_Q002-pending-platform.md`。
- **Q002 平台已改善**（可见点约 -7%～-30% vs 1345715），仍慢于榜最优。

### Q003（更大 tile）

- 条件分配：fp32 不建 scalesRaw；无 zp 不建 zp 缓冲；host 按 dtype/zp/indices 重算 UB 预算。
- **勿再把 `TQue` 模板深度改回 1**（须与 `InitBuffer(...,2)` 一致）。
- 止损：`decisions/DEC_Q003-pending-platform.md`；失败/无收益回退 Q002。

---

## 6. 给后续 AI 的优化队列（有界）

先读 `path-ledger.md`，**新开一路必须先记账**。

| ID | 假设 | 建议做法 | 止损 |
|----|------|----------|------|
| Q002 | 流水重叠 | updates 双缓冲；MTE2 与 V/MTE3 重叠 | 1 人日；无增益 <2% 写 DEC |
| Q003 | UB 吃满 | 收紧 per-elem 缓冲；fp32 scales 免 raw | 0.5 人日 |
| Q004 | 小 shape | 仿真图豁免，勿抠 us | ≤2h |
| Q005 | 拷贝占比 | profile 后：若 inplace 为主路径，拷贝优化收益有限 | 先看 msprof |

**禁止**：

- 无 `path-ledger` 新行就换路线  
- 未过 `-e` 精度就宣称性能达标  
- 改 `people/<他人>/` 或其它 `tasks/active/*`  
- 删除历史台账 / 假 PASS  

性能门禁总则：`docs/collaboration/02-perf-gate.md`。

---

## 7. 云测最小闭环

见 `CLOUD-RUN.md`。测完请：

1. 把完整 log 放进 `explorations/logs/EXP_Q001/`  
2. 更新 `path-ledger` Q001 结果行 + `dev-log`  
3. `git push` 到 `main`（公开仓，不 push = 队友看不见）

---

## 8. 会话卡片（下一位 AI 可复制）

| 字段 | 内容 |
|------|------|
| 题名 | `QuantUpdateScatter_2330` |
| STATUS | 性能中 |
| 唯一目标 | （例）云测 Q001 精度 + 记 profile；或开 Q002 双缓冲 |
| 角色 | Implementer |
| 允许路径 | `people/yuhongming-2026/QuantUpdateScatter_2330/**`；本题 `tasks/active/QuantUpdateScatter_2330/**` |
| 禁止 | 其它题；他人 people；无台账换路；假 PASS |

---

## 9. 已知未知

- 官方性能门禁 shape / TBE 对照表：**未入库**。拿到后放到 `materials/` 并改 STATUS。  
- 本机 Windows **无 NPU**，Q001 编译与精度未在本地验证。  
- `SyncAll` 在拷贝后仍需要（多核写 out 前）；不要轻易删。
