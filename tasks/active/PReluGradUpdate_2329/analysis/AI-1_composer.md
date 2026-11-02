# AI-1 优化分析 — PReluGradUpdate

> 分析方：本会话 Composer（协作仓主分析之一）  
> 材料：`materials/_unzipped/code`（来自 `PReluGradUpdate_problem_2329_template.zip`）  
> 阶段：仅分析，不实现

---

## 1. 题目理解（验收口径）

| 项 | 结论 |
|----|------|
| 语义 | 与 TBE/`reference.py` 对齐：`dx = f>0 ? g : w*g`；`update = f>0 ? 0 : f*g` |
| 广播 | `weights[C] → [1,C,1,…]`，**通道维对齐**，非右对齐 |
| update | **与 features 同形**，禁止按通道 reduce（模板注释已纠正旧 aclnn 误解） |
| 比较 | **逻辑比较** `features > 0`（含 =0 走负支），非二进制位模式比较 |
| dtype | fp16 / fp32 / bf16；参考实现 **fp32 计算再 cast** |
| 平台 | OpDef：`ascend910b`；Format：ND + AutoContiguous |
| 性能 | 全核场景 ≥ TBE **95%**；&lt;10us 且差 ≤3us → 仿真图+分析豁免 |
| 泛化 | 合法 shape/dtype 均需可算；验收用泛化数据 |

模板现状：Kernel/Tiling/InferShape 多为 TODO；`BUFFER_NUM=2` 已预留双缓冲；当前 `SetBlockDim(1)`，性能基线不存在。

---

## 2. 最简正确路径（必须先走通）

**主张：先做「单次遍历 + 逻辑 mask + 通道广播」的向量化实现，再谈性能。**

### 2.1 Host

1. **InferShape**：`dx`、`update` 均拷贝 `grads`/`features` 的 shape（二者须一致）；勿把 `update` 推成 `[C]`。  
2. **Tiling**：读取 shape → `totalNum = N*C*…`；按 AIV 核数切 `blockFactor`；按 UB 与同时驻留的 buffer 数切 `ubFactor`（对齐 32B/块）。  
3. **额外 tiling 字段建议**（模板仅 3 字段，**极可能不够**）：至少需要 `channelNum`、`innerNum`（C 后维度积）或等价信息，以便 Kernel 从线性下标还原通道；否则只能假定错误的右对齐广播。  
4. **TilingKey**：fp16→mode0，fp32→mode1；**bf16 不能无脑当 half**——需独立 key 或明确 `bfloat16_t` 模板实例（当前模板是风险点）。

### 2.2 Kernel 数据流（最简）

对每个 UB tile（已 AutoContiguous 的 ND）：

1. CopyIn：`grads`、`features` 同长 tile；`weights` 按本 tile 覆盖的通道 **gather/复制到 UB**（或按 `innerNum` 周期复用）。  
2. Compute（建议 fp16/bf16 升 fp32 算，再降回）：  
   - `mask = features > 0`（CompareScalar 逻辑比较）  
   - `dx = Select(mask, grads, grads * weights_b)`  
   - `update = Select(mask, 0, features * grads)`  
3. CopyOut：`dx`、`update`。  
4. `Process`：双缓冲流水（模板已声明 VECIN/VECOUT 双队列，沿用即可）。

**不必做**：update 的 atomic 通道求和、Cube、复杂 workspace（模板 `WS_SYS_SIZE=0` 合理）。

---

## 3. 性能路径（有界、最简优先）

本算子本质是 **逐元素 + 轻量广播**，通常 **内存带宽受限**。收益排序：

| 优先级 | 动作 | 预期 | 时限建议 |
|--------|------|------|----------|
| P0 | 功能精度全过（含 bf16、C=1、高维、f≤0 边界） | 可验收正确性 | 不设性能目标 |
| P1 | 全核均分 + 32B 对齐 + 双缓冲 | 逼近 TBE 全核带宽 | 0.5～1 人日 |
| P1b | **一次读 grads/features，同时产 dx/update**（减少 GM 往返） | 关键带宽优化 | 含在 P1 |
| P2 | 仅当全核 case &lt;95% TBE：查 UB 利用率、是否多余 cast、是否重复 Compare | profile 后再改 | ≤1 人日，触线写 DEC |
| B 类 | 小 shape &lt;10us | **仿真图+分析**，不抠 us | ≤2h |

### 明确不要先做（反模式）

- 未对齐语义就做多 tilingKey「炫技」分叉  
- 把 update 做成 reduce（与 TBE 冲突，必炸精度/用例）  
- 二进制/位级「比较优化」替代逻辑 `>0`  
- 无 profile 就上多轮 GM、手工汇编级调优  
- 为小 shape 死磕微秒（任务书已给豁免）

---

## 4. 本题特有风险

1. **通道广播 vs 平坦切分**：只按 `totalNum` 切核时，tile 可能跨通道边界 → 权重索引必须用 `c = (idx / inner) % C` 或按行处理。  
2. **bf16 模板缺口**：`MODE_0` 绑 `half`，bf16 走同一 key 会错。  
3. **InferShape 模板 bug 味**：循环用 `GetInputShape(i)` 填 output i；i=1 碰巧 features 同形，但校验不完整（未显式断言 grads↔features）。  
4. **零与负数**：`f>0` 为假时走 weight 支路；`f==0` 的 dx/update 须与 TBE 一致。  
5. **精度**：半精度比较+乘加顺序；以 Judge 默认阈值 + 参考 fp32 路径为准。  
6. **样例仅 float32 `[4,6,8]`**：自测必须自建 fp16/bf16/多 rank/泛化表，不能只靠 samples。

---

## 5. 建议道路（写入台账的候选）

| ID 建议 | 内容 | 结论倾向 |
|---------|------|----------|
| P010 | 补齐 InferShape + 语义单测（CPU ref） | 必做 |
| P011 | 扩展 TilingData（C、inner）+ 全核切分 | 必做 |
| P012 | Kernel：逻辑 Compare + Select + 双出 | 基线采纳 |
| P013 | bf16 独立模板/key | 必做 |
| P014 | 双缓冲 + 单遍产双输出 | 标准性能 |
| P015 | 小 shape 仿真豁免材料 | 按需 |
| — | update reduce / 位比较 / 过早多 key | **废弃** |

---

## 6. 有序行动清单

- **P0** 锁定语义与 shape/dtype（含 bf16、广播、逻辑比较）；修 InferShape；扩展 tiling 元数据。  
- **P1** 最简 Kernel 全核双缓冲，一次遍历写 dx+update；自建泛化用例。  
- **P2** 仅全核未达 95% 时 profile；小 shape 走仿真分析，停止死磕。
