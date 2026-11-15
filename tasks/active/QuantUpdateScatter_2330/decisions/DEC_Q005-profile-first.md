# DEC Q005 — 先补瓶颈证据，再改流水

日期：2026-11-15。承接 DEC_Q003，现行代码为 2d3c61c 中已采纳的 Q002。

- 唯一目标：提供可重复、逐 case 校验的 Q002 采集入口，补齐 kernel 时长 / PipeUtilization 证据；本轮不推测加速。
- 会话：Implementer，yuhongming-2026；仓库 D:/sageAttention2 算子开发任务书，main。
- 允许写：people/yuhongming-2026/QuantUpdateScatter_2330/**；tasks/active/QuantUpdateScatter_2330/**。
- 初版范围：仅本地工具；review 后已允许入库供云采（仍不改生产 host/kernel）。

## 已核实

Q003 已废弃，不能重开省 UB / 抬 tile。当前 TQue 深度 2，InitBuffer 两槽；不再改回 1。逐行 EnQue 后立即 DeQue 是源码事实，但设备指令异步，仅据此不能断言没有重叠。现存证据只有平台汇总和冒烟日志，没有 msprof 时间线、流水指标及隐藏 45 点详情。

## 本轮决策

新增独立 profiling runner / 数据准备 / 结果校验与归档入口，覆盖四 dtype 签名、可选 zp、reciprocal 和输入输出同址。保留默认样例、生产 host/kernel、SyncAll 和量化链。先生成小而明确的诊断 case，不把它们冒充官方 50 点。采集直接运行已构建的程序，不把编译包进 application，不同时传 --export。逐次重新获取 executor，避免未经开启复用的 executor 被重复执行。

## 止损与后续

限 0.5 人日；本地验证只记录工具和 CPU 检查，不记设备 PASS。没有同机原始 profile 不实施下一条生产性能候选。拿到数据后比较 inplace / out-of-place、单行 / 多行、宽 / 窄行：若复制占主导再立复制路线；若 MTE2 / V 等待明显再立跨行预取路线；若主要是固定开销则记录小 shape 限制。