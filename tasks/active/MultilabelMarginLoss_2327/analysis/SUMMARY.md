# 汇总 — MultilabelMarginLoss_2327

| 项 | 内容 |
|----|------|
| 状态 | **已定稿；实现中** |
| Cursor 三路 | [AI-1_composer.md](AI-1_composer.md) · [AI-2_grok.md](AI-2_grok.md) · [AI-3_grok.md](AI-3_grok.md) |
| Codex/Claude | [AI-codex-claude_design_final.md](AI-codex-claude_design_final.md)（实施细则优先） |
| 队友 | yuhongming baseline `people/yuhongming-2026/MultilabelMarginLoss_2327/` |
| 本仓初版 | `people/hongwei-2026/MultilabelMarginLoss_2327/`（baseline + classTile） |
| 隔离 | 仅本题；不采用其他题门禁数字当已生效事实 |

---

## 共识（三路一致）

1. **金标准**：`samples/reference.py`（对齐 910B TBE）。冲突时压过 PyTorch/aclnn「0/1 is_target」叙述。  
2. **语义**：遇首个 `<0` 截断前缀；`is_target` = **出现次数（int32，形 `[N,C]`）**；  
   `losses[i]=(1/C)*Σ_{j:cnt=0}Σ_{l:cnt>0} cnt[l]*max(0,1+x[j]-x[l])`；无正类则 0 且仍计入 mean 的 N。  
3. **形状**：`none→y[N]`；`mean/sum→y[1]`（非 0 维）；`K` 可≠`C`。模板 InferShape **必须重写**。  
4. **精度链**：中间 **fp32** 累加，最后再 cast；禁止半精度算 hinge；bf16 **禁止**走 `half` 键。  
5. **并行粒度**：默认 **按行分核**；**禁止按 K 分核计数**（截断前缀不可拆）。  
6. **大 C**：pos 常驻 + neg 滑窗（或 j×l 分块）；**禁止** UB 展开整张 `C×C`。  
7. **归约**：mean/sum 经 **fp32 workspace**，**单一核有序 Finalize** 写 `y[0]`；none 下行主核独占写 `y[i]`（窄写/Pad，勿 32B 踩邻行）。  
8. **性能**：正式 95%/小 shape 条文 **待赛题核对**；未核对前不对 `[4,6]` 抠 µs；须同机 TBE 对照才开性能专题。  
9. **工程**：扩展 TilingData（`N,C,K,reduction,cores,ubTileC…`）；三 dtype 分键；改 UT 期望，勿为保留 `"0 1 0 "` 留空 tiling。

---

## 分歧与裁决

| 点 | AI-1 | AI-2 | AI-3 | **裁决** |
|----|------|------|------|----------|
| mean/sum 多核写回 | 原子加 **或** core0 单写 | **按核序** workspace 收；否定半精度原子抢写 | **禁止**无序 AtomicAdd 到 `y[0]`；优先长度 **N** 的 `losses[]` 再有序 reduce | **采纳 AI-2/3**：P0/P1 用 fp32 `losses[N]`（或等价）+ 单核 Finalize；AtomicAdd→`y[0]` 进废弃清单 |
| P0 是否多核 | 按 N 分核 | 按行切核（可多核） | 可先单核 `workspace=0` | **P0 允许单核打通**；若实现成本低可同步按 N 分核，但 Finalize/窄写规则不变 |
| 与 numpy64 对拍 | 未细拆 | 大 C 不在 P0 模拟 fp64 | 应对齐 **TBE fp32**；大 C 用同机 TBE 仲裁 | **采纳 AI-3**：实现对齐 TBE fp32；大 C 不以 numpy64 bit-exact 强改算法 |
| 1/(N·C) 一次除 | 未强调 | mean 只再 `/N` | 禁止未对拍就合并 | **禁止**融合成 `Σ/(N·C)` 除非 DEC+对拍 |

---

## 执行顺序

| 级 | 内容 |
|----|------|
| **P0** | InferShape；三键；TilingData；逐行截断→计数→fp32 hinge→写 `is_target`/`y`；大 C 至少 neg 分块不炸 UB；`-e` + 扩泛化对拍 reference（含 K≠C、截断、重复、空正类、三 reduction） |
| **P1** | 按 N 全核；workspace `losses[N]` + Finalize；正类固定 + 负类向量化 hinge；双缓冲 |
| **P2** | 仅门禁未达标且有同机 TBE 表：pair 空间分核（仍单写 `y[i]`）；累加顺序对齐；触碰止损写 DEC |

---

## 废弃清单（无新证据不得重开）

- `is_target` 做成 0/1；按文档/PyTorch 当 golden  
- 按 **K** 分核并行计数 / 拆前缀  
- InferShape 维持 `y←x`、`is_target←target`  
- bf16→`half`；半精度算 hinge  
- UB 展开整表 `C×C`  
- 多核无序 `AtomicAdd` 到 `y[0]`；各核先 mean 再平均  
- `none` 下整 block `DataCopy` 踩邻行 `y`  
- 未对拍的 `Σpairs/(N·C)`  
- 整段粘历史 ops-nn AscendC 当已验收  
- 串用其他题 95%/Select/Compare 结论当本题事实  
- 为小 shape 占核去切 C；未 SUMMARY 就开 P2  

---

## 回写

- `path-ledger`：P002 结案 → 采纳本 SUMMARY；下一闸升格 `design.md`  
- 实现落点：须声明登录名后仅 `people/<成员>/MultilabelMarginLoss_2327/`  
