# DEC Q006 — 跨行预取与递增寻址候选

日期：2026-11-15。Implementer：yuhongming-2026 / Codex。单一目标：实现并验证 Q006，减少 Q002 多行路径的调度与地址计算开销。允许路径仅本题 seat 与 tasks；不重开 Q003，不调整整表 copy/UB/tile。

## 证据和授权

Q005 已设备结案：narrow_rows_in 457.42us、wide_rows_in 40.43us，scalar 开销显著；大 var 异址 copy 税不能解释全部更新路径。来源 logs/EXP_Q005/rows01-medians.json、copy01-medians.json。用户明确批准 Q006 1 人日候选、云测及 push main。此立项不声称已有时间线证明 Q002 没有异步重叠。

## 具体候选

- 保留 TQue 模板深度 2、InitBuffer 两槽和全部现有 UB 分配。
- 拆分 CopyIn 与 Compute/CopyOut：每 batch/column 先入队首行；每次 DeQue 当前行之前入队下一行。最多两行待消费。输入行在同一 batch 的 middle/update 行之间连续，允许跨 middle 预取；batch 结束全部排空。
- 每 batch 读取一次 index，更新区的 src/dst 起点提前计算。逐行只递增 lastDim；middle 结束按 varAxis 与 updateAxis 差距跳过未更新行。不按行做除法/取模或重新计算完整乘积。
- 单行 direct 路径保留；copy、SyncAll、参数 tile 复用、四 dtype 签名、fp32→int16(CAST_RINT)→half→int8 不变。indices cap/fallback 本轮不另开路线。

## 风险与验证

队列满/空、末行多读、middle 跳步、短 indices、跨 column 参数覆盖和共享 scratch 依赖是重点。先保存精确 Q002 源码及 hash，再用实际类 CPU 对拍 Q002/标量 oracle；增加异步队列依赖模型检查和提交顺序检查。CPU 模型不能证明 NPU 真实重叠或 CANN 接受代码。

## 门禁和止损

限 1 人日。云 -e mismatched=0 后才进入平台/诊断性能比较；50 点精度和 Q002 五个可见点仍是外部门禁。任意精度失败或相对 Q002 可见点增益 <2%，回退生产 Q002，保留候选/失败日志及 DEC。不采纳未测候选为性能基线；设备连接缺失时如实记录 NOT_RUN，提供具体云测入口，不伪造 PASS。
## 本地实施结果

Q006 候选已实现；3072 组对拍通过；队列深度仍 2。

## Cursor review（同日）

采纳为**待云测候选**并允许入库：TQue=2 正确；`else` 仅单行 direct；未改 host/RINT/SyncAll/四签名；输出侧仍即入即出（本轮只做输入预取）。无设备数字前不升级采纳基线。云 `-e` 失败或可见点增益&lt;2% → 回退 `explorations/Q006/baseline/`。
