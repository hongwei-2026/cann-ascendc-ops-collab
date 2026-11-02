# 汇总 — PReluGradUpdate（已定稿）

| 项 | 内容 |
|----|------|
| 状态 | **已定稿**（可进入设计；仍禁止无台账的性能死磕） |
| 来源 | [AI-1_composer.md](AI-1_composer.md) · [AI-2_grok.md](AI-2_grok.md) · [AI-3_grok.md](AI-3_grok.md) |
| 定稿日 | 2026-03-22 |
| 隔离 | 仅本题；不得套用到 SpMV / SageAttention2 等 |

---

## 共识（三路一致，必须执行）

1. **语义**：`dx = f>0 ? g : w*g`；`update = f>0 ? 0 : f*g`；`weights` 按 **dim1 通道** 广播；`update` **同形不 reduce**。  
2. **比较**：逻辑 `GT`；禁止位型/整数比较；`==0`、`-0`、NaN 走 else（与 reference/TBE 一致）。  
3. **画像**：带宽受限逐元素算子；workspace=0；不上 Cube。  
4. **bf16**：必须独立 tiling key / `bfloat16_t`，禁止复用 `half`。  
5. **TilingData**：仅有 `totalNum/blockFactor/ubFactor` **不够**，必须能表达 `C`、`inner`（及切分模式）。  
6. **实现顺序**：InferShape/校验 → 通道正确 tiling → 单趟融合 Kernel → 泛化矩阵绿 → 全核对照；小 shape 走仿真豁免。  
7. **反模式（废弃，无新证据不得重开）**：update 通道求和；权重 expand 到 GM；TBE 的 0/1 中间张量落 GM；假全核硬铺小 shape；无 profile 抠微秒。

---

## 分歧与裁决

| 点 | AI-1 | AI-2 | AI-3 | **裁决** |
|----|------|------|------|----------|
| 切分主策略 | 平坦元素 + 下标还原 `c` | **行片 ROW_SLICE / PACK_ROWS**，禁止跨通道 | 按通道连续扫 inner，勿 flatten round-robin | **采纳 AI-2 为主**：以不跨通道的行片/整行打包切分；允许用 `c=(idx/inner)%C` 仅作校验或退化路径，不作主路径 |
| 流量口径 | 强调少 GM 往返 | 主张约 **4×** 主张量（2 读 2 写，权忽略） | 理想约 **5×**（含权 gather 上限） | **按「不物化权重」目标对齐 AI-2**：大张量 2 读 2 写；权重量级忽略；禁止为权再加 1× GM |
| 双缓冲 | 模板双队列沿用 | **四队列**（grads/features/dx/update）深度 2 | 5 路账 + tile≥2 才开双缓冲 | **采纳四队列双缓冲**；小 shape / 单 tile 可不强开 |
| 计算顺序表达 | Compare + Select + Mul | 一次 GT + 标量 Muls + 两次 Select，UB 内 fp32 | 同左，强调单趟双出 | **一致采纳**：UB 内升 fp32 计算再 cast；单趟产 dx+update |
| 测试强调 | 泛化自建 | 正/负/0/NaN 极性探针 | 完整 F/Q/B/N 矩阵 + 零值探针 | **合并采纳**：以 AI-3 矩阵为检查表，AI-2 极性用例为必测 |

---

## 执行顺序（回写台账）

| 级 | 内容 | 时限意识 |
|----|------|----------|
| **P0** | 修 InferShape（两输出=features；校验 rank≥2、同形、`weights=[C]`）；扩展 TilingData；行片切分 + 全核策略；Kernel 单趟逻辑比较融合；bf16=MODE_2；泛化/极性/C=1/高 rank 用例绿 | 功能基线；亦为性能基线预期 |
| **P1** | 仅当全核大 shape &lt;95% TBE：查核数、预取重叠、32B 对齐、ubFactor；增益&lt;2% → DEC 停 | ≤1 人日 |
| **P2** | &lt;10us 且差≤3us：仿真图+流量说明（≤2h）；无向量侧堵 trace 不做指令级融合 | 默认不做深度优化 |

---

## 废弃清单（禁止重复开工）

- update 做成 `[C]` 或 atomic 通道归约  
- 二进制/符号位比较替代逻辑 `>0`  
- 权重 broadcast 成同形 GM / 非零 workspace  
- bf16 走 `half` kernel  
- 为对齐 DMA 让 tile **跨通道** 却共用一个标量 w  
- 拿样例 `[4,6,8]` 当全核 95% 调优对象  
- Cube / NZ / 多套小 shape 微核  

---

## 设计入口（下一步）

依据本 SUMMARY 撰写本题 `design.md`（仍在本目录，勿写入其他题）。  
台账对应：`path-ledger.md` 中 P003 结案为采纳；P004 按本页 P0 启动。
