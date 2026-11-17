# 汇总 — SyncBatchNormGatherStats_2332（合并定稿）

| 项 | 内容 |
|----|------|
| 状态 | **已定稿（可进入设计/实现；深度改 Kernel 须用户确认）** |
| 三路 | [AI-1_composer.md](AI-1_composer.md) · [AI-2_grok.md](AI-2_grok.md) · [AI-3_grok.md](AI-3_grok.md) |
| 隔离 | 仅本题；禁止套用 QuantUpdateScatter / PRelu 等数字与惯性 |
| 座位代码 | `people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/` |

---

## 共识（三路一致）

1. **权威语义**：`samples/reference.py`（对齐 910B 内置 + 图上 Cast）。不是教科书纯 fp32 BN。
2. **精度铁律**：输入先 fp16 量化 → 中间 fp32 → 四输出再落 fp16 网格；`count==1` → `n/(n-1)=+inf`，勿修有限；`new_mean`/`new_variance` OpDef 恒 float。
3. **基线已是「最简正确实现」结构**：ZIP Host/Kernel 已含 QuantFp16Vec、world 累加、inf、invstd NaN/inf 标量补丁、`kMinBlock=8`、两档 tiling key。P0 = **验证 + 补测**，不是重写。
4. **samples 首点 `[1,6]` / C=6**：只做正确性冒烟；**禁止**当性能结论。
5. **性能专题未开门**：功能/精度未云测结案、无本题 TBE 对照表前，禁止双缓冲/事件重叠/改 min-block。
6. **并行维是 C，不是 world**：设备维是归约；勿一通道一核；勿默认 world 切核 + workspace。

---

## 分歧与裁决

| 点 | AI-1 | AI-2 | AI-3 | **裁决** |
|----|------|------|------|----------|
| `sample_count` 多元素 | Kernel 求和 vs reference 取 `[0]` | Host 保持透传 numel，等官方用例 | 同左；性能非热点 | **P0 跟 Kernel：求和**；补 `W>1`+向量 count 用例钉死；未证实前禁止把 countNum 折成 1 |
| `world_size>1` 与 reference | reference `reshape(-1)` 不适多 device | Host 折叠前导维正确 | Kernel 循环正确；W=1 才与 reference 数值等价 | **采 Kernel 跨 device 累加**；扩展 reference 或对设备 bin；勿删 world 循环 |
| `new_var` fp64 vs Kernel fp32 | 可能 1 ulp | — | 以 bin 为准 | **以 Judge/内置 bin 为准**；勿为追 fp64 改语义 |
| fp16 store 二次 Cast 吞 NaN | 标量补丁在 Quant 后 | — | half store 再 CAST_RINT 可能再吞 | **P0 必测**负 var+eps + fp16 路径；FAIL 再最小修（补丁挪到最终 Cast 后） |
| 性能第一刀 | 勿先做 | Host 先不大改切分 | K0 只测量 → K1 事件重叠 | **先云 `-e` 与精度结案**；过闸后 K0 对照，再单列 K1 |

---

## 执行顺序

| 级 | 内容 |
|----|------|
| **P0** | 用户确认后：云 `build.sh -j8 -e` + 全文 log 归档；对照 reference/内置：普通点、`count==1`、负 var+eps、Test_050 类量化、四组 dtype、（若有）`W>1`；Host UT 补 C=1/8/9、int64 count；**不改切分公式** |
| **P1** | 功能精度台账结案 + 抄齐本题任务书性能表 + 同机 TBE 对照后：K0 测量 →（仅 A 类未达标）有界 K1 事件重叠 / ub 帽；一次一条 path-ledger |
| **P2** | 仅 DEC 后：world 软流水（W≫1）、向量谓词替标量环（大 tile）、结构性双缓冲；小 shape 走仿真豁免，不抠 us |

---

## 废弃清单（无新证据不得重开）

| ID | 方向 |
|----|------|
| D-01 | 纯 fp32、跳过输入/输出 fp16 量化 |
| D-02 | `count==1` 无偏系数改有限 / clamp |
| D-03 | 删 invstd 标量 NaN/inf 补丁（或挪到最终 Cast 前） |
| D-04 | 用 case.json C=6 做性能结论或为其调 min-block/双缓冲 |
| D-05 | 一通道一核 / 弃 `kMinBlock=8` 刷占用 |
| D-06 | world 切核 + workspace 跨核 Add（默认可废弃） |
| D-07 | 为 count 宽度 / n==1 / N==1 膨胀 tiling key |
| D-08 | P0 把 countNum 恒置 1 贴合 reference `[0]` |
| D-09 | 串题套用他题 95%/10us/双缓冲结论 |
| D-10 | 功能未绿就改流水线 / 捆改 K1+K2+K3 |

---

## 回写 path-ledger

- **S001**：开题脚手架 + 三路分析 → **采纳（开题）**
- **S002**：P0 云冒烟 + 精度边界（用户确认后开工）→ 规划中

---

## 下一闸

等用户明确「可以编码 / 可以云测」→ 执行 P0；本会话到此 **不深度改 Kernel、不开性能专题**。
