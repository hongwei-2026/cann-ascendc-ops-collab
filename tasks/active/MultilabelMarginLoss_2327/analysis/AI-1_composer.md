# AI-1 分析 — MultilabelMarginLoss_2327

- 来源：Composer（本会话）
- 日期：2026-11-03
- 材料：`materials/_unzipped/code/`（`reference.py` / OpDef / 占位 Host·Kernel）
- 范围：语义理解、最简正确路径、性能候选与反模式；**不编码**

---

## 1. 语义与验收理解

### 1.1 金标准

`samples/reference.py` 明确对齐 **910B TBE**（`ops_legacy/dynamic/multilabel_margin_loss.py`），并写明与 PyTorch/aclnn 文档有两处实质差异：

1. **`is_target` 是出现次数（int32）**，不是 0/1。每行从左扫描 `target`，遇**首个 `<0`** 截断；对取值 `t∈[0,C)` 做 `bincount`。
2. **loss 按次数加权**：  
   `losses[i] = 1/C * Σ_{j:cnt=0} Σ_{l:cnt>0} cnt[l] * max(0, 1+x[j]-x[l])`

中间量 **float32**，最后按输出 dtype 舍入。`mean`/`sum` 输出必须是 **shape `[1]`** 的 ndarray（判题机拒 0 维标量）。

### 1.2 接口（模板 OpDef）

| 参数 | 角色 | dtype | 形状要点 |
|------|------|-------|----------|
| `x` | 入 | fp32/fp16/bf16 | `[N,C]` |
| `target` | 入 | int32（三档同绑） | `[N,K]`，K 常=C，可≠C |
| `y` | 出 | 同 `x` | none→`[N]`；mean/sum→`[1]` |
| `is_target` | 出 | **恒 int32** | `[N,C]`（跟 `x` 的 C，不是跟 K） |
| `reduction` | attr | String | `"none"`/`"mean"`/`"sum"`，默认 `"mean"` |

平台：`ascend910b`。首测点 `case.json`：`[4,6]` + `mean`。

### 1.3 验收门禁（已见 / 待核对）

- 样例：`y` rtol/atol `1e-4`；`is_target` 整数全等。  
- 性能：同系列 2329 惯例为「全核 ≥ TBE 95% + 小 shape 豁免」——**须用正式赛题条文核对后再写入门禁常量**。  
- 小 shape 豁免是否适用：待核对；若适用则走本仓 `02-perf-gate` A/B 分流。

---

## 2. 模板现状与缺陷

| 模块 | 现状 | 风险 |
|------|------|------|
| `*_def.cpp` | 2 入 2 出 + String reduction，dtype 绑定合理 | `is_target` 已锁 int32，正确 |
| `*_infershape.cpp` | 输出直接拷输入 shape | **错误**：`y` 在 mean/sum 应为 `[1]`，none 为 `[N]`；`is_target` 应为 `[N,C]` 而非盲目拷 `target` |
| `*_tiling.cpp` | 占位 `totalNum=0`，`blockDim=1`，workspace=0 | 不可用 |
| tilingKey | fp16 **与 bf16 同走 MODE_0 → `half`** | **静默错误**（与 2329 同类坑，但结论属本题独立确认） |
| Kernel | 空 TODO；Que 只声明了 x/y | 缺 target / is_target / workspace |
| TilingData | 仅 3 字段 | 需扩：`N,C,K,reduction,core…` |

---

## 3. 最简正确实现路径（P0）

**目标**：先过功能+精度；不做炫技。

1. **InferShape**：读 `reduction`；校验 `x`/`target` rank=2、`N` 一致；`y`/`is_target` 按上表设形；非法返回 `GRAPH_FAILED`。  
2. **Tiling**：按 **行（N）** 分核；每核连续行区间；`blockDim = min(coreNum, N)`（N=0 特判）；扩展 TilingData；workspace：mean/sum 跨核归约缓冲（float partials，按核或按行）。  
3. **TilingKey**：至少 **3 键**——fp16 / fp32 / bf16 分离；bf16 升 fp32 计算再 RNE 回写（或专用路径，禁止当 `half`）。  
4. **Kernel（逐行）**：  
   - CopyIn 一行 `x[C]` + `target[K]`；  
   - 标量扫 target → 写 `is_target` 一行（int32）；  
   - 用 float32 累加 pairwise hinge；除以 `C`；  
   - none：写 `y[i]`；mean/sum：写 workspace partial，core0 Finalize。  
5. **小 C / 小 N**：允许标量双重循环；正确性优先。

复杂度：每行最坏 `O(C²)`（或 `O(#neg * #pos)`）。大 C（如参考注释 `[512,4096]`）是后续性能主战场，**P0 可不向量化**，但 Host 须能 tiling 这类 shape 而不炸 UB（按行/按块切）。

---

## 4. 性能候选（有序，有止损）

| 优先级 | 方向 | 假设 | 止损 |
|--------|------|------|------|
| P1 | 多核按 N 均分 + 全核参与 | 验收强调「所有核参与」 | 若 N≪coreNum，记录小 shape 豁免，勿死磕 |
| P1 | 固定 `l∈正类`，对 `j` 向量化 `max(0,1+x[j]-x[l])` 再 `Muls(cnt[l])` 归约 | C 大时有效 | 正类很少时收益有限；换路先记账 |
| P2 | 双缓冲 / UB 内多行 | 带宽够 | UB 不够则退回单行 |
| P2 | 预建 pos/neg 索引压缩内层 | 稀疏标签 | 实现复杂，P0 后视 profiling |
| 废弃候选 | 整段照搬已删除的 ops-nn AscendC 当「已过验收」 | 上游曾因兼容回退 | **禁止假 PASS**；可借思路但须对本题 golden 重测 |

---

## 5. 反模式（不要先做）

1. 按文档把 `is_target` 写成 0/1。  
2. 用「截断前所有位置展开」却忘记与 `cnt[l]` 加权等价时的边界（非法类 id、截断后垃圾）。  
3. fp16/bf16 共用 `half` kernel。  
4. mean/sum 多核直接写同一 `y[0]`（32B RMW 竞态）——须原子累加或 core0 单写。  
5. 未过 SUMMARY 就开 P2 大 C 极限优化。  
6. 把 2329 的 `Compare+Select` 故事直接贴进本题（本题核心是 **计数 + 双层 hinge**，不是 PReLU 分支）。

---

## 6. 本题特有风险清单

- **K≠C**：`is_target` 宽为 C；target 越界 id 忽略。  
- **重复标签 / 无正类**：计数与 `losses=0`。  
- **非有限值**：`1+x[j]-x[l]` 与 `max`；中间 float32；与 TBE/golden 对拍 Inf/NaN（实现前用 numpy 扩用例）。  
- **reduction 字符串解析**：与仓内 loss 算子一致用 `GetAttrPointer<char>`，勿当 int。  
- **workspace**：`GetUserWorkspace` 与系统预留分离（ops-nn 历史 PR 教训可参考，但须本机复测）。

---

## 7. 建议执行顺序

```text
P0  InferShape + 三键 Tiling + 逐行正确 Kernel → -e 首点 + 泛化对拍 reference
P1  全核分 N + mean/sum 安全归约 → 平台功能/精度
P2  大 C 向量化 / profiling → 仅门禁未达标再开；触碰止损写 DEC
```

---

## 8. 回写台账建议

- 采纳：TBE/reference 计数语义；按行分核；三 dtype 分键；mean/sum 禁止多核裸写标量。  
- 待 SUMMARY：大 C 向量化细节、workspace 布局、是否需要单独 schMode（none vs reduce）。
