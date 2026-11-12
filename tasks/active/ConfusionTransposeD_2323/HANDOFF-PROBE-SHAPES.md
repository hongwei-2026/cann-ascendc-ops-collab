# HANDOFF — ConfusionTransposeD_2323：问题总结 + 探针探形（给其他 AI）

> 日期：2026-11-12  
> 题名：**ConfusionTransposeD_2323**  
> 活动代码座位：`people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/`（P004 / commit `4761239`）  
> 共享台账：`tasks/active/ConfusionTransposeD_2323/`  
> 围栏：只改本题；代码只写 `people/<本成员>/`；先记账再换路；**未立项禁止大改 Kernel**。

---

## 1. 一句话

平台 1～5 **结果对**（错误 0%），但比最优慢约 **200～600×**；整次仍 **Runtime Error**（隐藏点 6 未公开）。同机系统 TBE 已测到 **~4.8μs**，与榜首 ~3.5μs 同阶。下一步优先用**探针拿到每个 case（含隐藏）的真实 shape/perm/attrs**，再决定是否镜像 TBE 模板。

---

## 2. 最新平台成绩（P004 再提）

| 测试点 | 结果 | 错误 | 用时 | 最优 | 约慢 |
|--------|------|------|------|------|------|
| 1 | Pass | 0% | **716.88μs** | 3.58μs | ~200× |
| 2 | Pass | 0% | **2.05ms** | 3.45μs | ~590× |
| 3 | Pass | 0% | **2.03ms** | 3.43μs | ~590× |
| 4 | Pass | 0% | **724.61μs** | 3.50μs | ~207× |
| 5 | Pass | 0% | **719.96μs** | 3.42μs | ~210× |
| 6 | **隐藏** | * | * | * | 整次 **Runtime Error** 高度怀疑在此 |

提交状态：**Runtime Error**（不是 Wrong Answer）。  
活动包：P004 连续行 DMA（`4761239`），非 TBE 级实现。

模式提示（未证实，供探针假设）：

- **快档 ~720μs**：点 1 / 4 / 5 → 可能命中 P004「连续行」路径  
- **慢档 ~2.0ms**：点 2 / 3 → 可能走标量 gather / 非连续末维 / 更大或更碎布局  
- **隐藏点 6**：历史 C005 本地曾复现 Host `ACL561002`（整输入超 UB、perm/shape rank、shape 元素总数等），**不能**断定平台 RE 就是这些，但探针应覆盖

---

## 3. 已证实事实（勿重复踩坑）

### 3.1 功能

- 样例 + 泛化：EXP_C001/C002/C004 本机 bitwise 多轮 PASS（含 bf16/int64 等）。  
- 平台点 1～5 输出正确（0% 误差）。

### 3.2 性能对照（P008 / DEC007）

| 项 | 结论 |
|----|------|
| 系统 API | `aclnnConfusionTranspose`（**无 D**）参数含 `perm, shape, transpose_first`；`libopapi` 可链 |
| torch 入口 | `torch_npu.npu_confusion_transpose`，`ASCEND_CUSTOM_OPP_PATH` **必须 unset** |
| Host 计时 | 四档 shape 全贴死 ~0.09ms → **废**，只能看 msprof device |
| 系统 device | 核名 `te_confusiontransposed_*`；OP Type=`ConfusionTransposeD`；`(2,1024,128)→(2,128,1024)` f32 **last10 mean=4.804μs** |
| 榜首 | ~3.4–3.6μs → 与系统 TBE **同阶**，不是玄学 |
| P004 微优化 | P006/P007 已止损回退；相对 TBE 是**结构性差距**，再抠 DMA 无授权 |

证据目录：`explorations/logs/EXP_C008/`（含 `C008-msprof-system-paste.log`）。  
决策：[decisions/DEC007-p008-tbe-device-baseline.md](decisions/DEC007-p008-tbe-device-baseline.md)

### 3.3 TBE 实现结构（借鉴线索，≠可直接 `#include`）

设备入口 `confusion_transpose_d_apt.cpp` 按 tiling key 分派：

- `TransposeTensorMove` / `TransposeSmallShape` / `TransposeCutOneAxis` / `TransposeCutTwoAxis` / `TransposeBigDim` / `TransposeNLast`  
- 路径在 CANN：`.../tbe/impl/ops_math/ascendc/transpose/arch35/`

座位 `examples/run.sh` 会**重新挂自定义 OPP**且调用 `aclnnConfusionTransposeD`（带 D）→ **不能**当系统基线。

### 3.4 平台 RE（未结案）

- 多次重提整次 Runtime Error；点 6 始终隐藏。  
- C005：本机扩测部分 case 在 **GetWorkspaceSize** 即 `ACL561002`，未 launch；与平台是否同一原因**未知**。  
- **禁止**用本机 22/22 宣称平台验收通过。

---

## 4. 本次交给你的唯一目标（Owner 指定）

**用探针探测每一个平台 case 的具体形状（含隐藏点），落盘可复现证据。**

本会话不要：

- 假 PASS / 删历史台账  
- 未探明 shape 就开新性能 Kernel  
- 改其他题目 / 他人 `people/`  
- 只口头结论不写 `path-ledger` / EXP / logs

允许：

- 只读 Host/Kernel 加**诊断探针**（打印或写文件：`in_shape, perm, shape, transpose_first, dtype, numel, 是否过 Host 检查`）  
- CodeArts / 平台侧能跑的最小复现  
- 探针结果归档到 `tasks/active/ConfusionTransposeD_2323/explorations/`

---

## 5. 探针设计要求（必须）

对**每一次**算子调用（平台若可插桩；或本地模拟平台 harness）记录：

1. `dtype`  
2. 输入 `x` 的 rank 与各维（含 0/-1 若出现）  
3. 属性 `perm[]`、`shape[]`、`transpose_first`  
4. Infer 后的最终输出 shape  
5. Host 是否拒绝（错误码/信息全文）  
6. 若 launch：可选 msprof / 自打时间戳（device 优先）  
7. 与 P004 路径判定相关的内部标志（若能安全打印）：如是否走 `copySpan` 连续行、UB 是否装下整输入等

输出格式建议（每 case 一行 JSON 或固定 CSV），例如：

```json
{"case_id":"?","dtype":"float32","in_shape":[...],"perm":[...],"attr_shape":[...],"transpose_first":true,"out_shape":[...],"host_ok":true,"path_hint":"contiguous_row|scalar|reject","err":null}
```

隐藏点：平台不给 id 时，用**调用序号 / 时间戳 / workspace 哈希**区分；务必保留完整日志。

---

## 6. 推荐工作步骤

1. 读：`STATUS.md`、`path-ledger.md`、`DEC007`、`EXP_C008*`、`EXP_C005*`（若在）、P004 Kernel `op_kernel/confusion_transpose_d.h` + Host tiling。  
2. 在 **Host tiling / GetWorkspaceSize 入口** 加探针（比 Kernel 更早，能抓住 RE）。  
3. 本地先用已知回归 case 验证探针不破坏 bitwise。  
4. 打包含探针的包上平台或 CodeArts 等价 harness，**回收 6 次调用的 shape 表**。  
5. 对照系统 TBE：对探到的每个 shape 跑 `npu_confusion_transpose` + msprof，填「P004 vs TBE」表。  
6. 有完整 shape 表后，再向 Owner 申请：镜像 TBE 模板结构大改 **或** 仅修隐藏点 RE。

预算建议：探针实现+本地验证 ≤2h；平台/云回传分析 ≤1h；**超时写 DEC，不盲改 Kernel**。

---

## 7. 关键路径

| 用途 | 路径 |
|------|------|
| 活动源码 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| Host | `.../op_host/confusion_transpose_d_*.cpp` |
| Kernel | `.../op_kernel/confusion_transpose_d.{h,cpp}` |
| 台账 | `tasks/active/ConfusionTransposeD_2323/` |
| P008 证据 | `.../explorations/logs/EXP_C008/` |
| 性能门禁 | `docs/collaboration/02-perf-gate.md` |
| 协作协议 | `docs/collaboration/05-ai-work-protocol.md`、`AGENTS.md` |

---

## 8. 成功标准（本交接）

- [ ] 点 1～5（及隐藏点）各有一份可复核的 shape/attr 记录  
- [ ] Runtime Error 对应到具体调用与 Host/Kernel 阶段（若能）  
- [ ] 每个已探 shape 有系统 TBE device 时间或明确「无法 profile」原因  
- [ ] `path-ledger` + EXP + logs 已更新并 **push main**

**非本交接成功标准：** 立刻追平 3.5μs、或未经 shape 表的模板大改。

---

## 9. 给接盘 AI 的开场提示词（可复制）

```text
题名 ConfusionTransposeD_2323；成员按 CLAIM 使用自己的 people/<login>/。
唯一目标：为平台 case（含隐藏）做 Host 探针，拿到 in_shape/perm/shape/transpose_first/dtype，定位 Runtime Error 阶段；不改性能算法、不镜像 TBE，除非 Owner 另批。
先读 tasks/active/ConfusionTransposeD_2323/HANDOFF-PROBE-SHAPES.md 与 DEC007、STATUS、path-ledger。
活动 Kernel 基线 P004=4761239。系统 TBE 同机 last10 mean=4.804μs@(2,1024,128)。
结束后写 path-ledger + EXP + explorations/logs，并提醒 push。
```
