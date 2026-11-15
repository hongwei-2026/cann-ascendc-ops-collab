# 道路台账 — QuantUpdateScatter_2330

> 规范见 `docs/collaboration/01-path-ledger.md`。开新路前先搜本表。

| ID | 日期 | 作者 | 阶段 | 假设 | 方法 | 时限 | 结果 | 结论 | 耗时 |
|----|------|------|------|------|------|------|------|------|------|
| P000 | 2026-11-15 | yuhongming-2026 | 实现 | ZIP 朴素实现已功能可编 | 导入 `people/.../workdir/code`（去 build） | 0.5h | 已导入 | **采纳（功能基线）** | 0.5h |
| Q001 | 2026-11-15 | yuhongming-2026 | 性能 | 拷贝单核+每行重载 scales/zp 是大头 | ①多核 var 拷贝 ②indices UB ③scales/zp 按 tile 复用；tiling 按 totalNum 抬核 | 1 人日 | 待云测 | **进行中** | — |
| Q002 | — | — | 性能 | 双缓冲可叠 MTE2/V/MTE3 | Ping-pong UB + 事件重叠 | 1 人日 | 未开 | 排队（等 Q001 数字） | — |
| Q003 | — | — | 性能 | 大 lastDim 可再抬 ubFactor | 收紧 kBufBytes / 去掉 raw 缓冲特化路径 | 0.5 人日 | 未开 | 排队 | — |
| Q004 | — | — | 性能 | 小 shape 豁免 | 仿真图 + 分析，勿死磕 | 2h | 未开 | 排队 | — |

## 废弃清单

| 项 | 原因 |
|----|------|
| fp32→half→int8 直接量化 | 与内置二次舍入不一致，精度风险（reference 注释已钉死） |
| 跳过 SyncAll 后立刻多核写 | 与并行拷贝竞态；除非证明 inplace 且无拷贝 |

## 下一闸

1. 云端 `-e` 冒烟  
2. 补门禁 shape + TBE 对照  
3. 有数字后再开 Q002
