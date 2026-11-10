# Path ledger — ConfusionTransposeD_2323

| ID | 日期 | 路径 / 假设 | 结论 | 证据 |
|----|------|-------------|------|------|
| P001 | 2026-11-04 | UB 常驻整输入 + 输出下标反查 src；dtype 按字节宽选 tilingKey 0/1/2/3 | **采纳**（样例 -e PASS 0/120） | EXP_C001 |
| P002 | 2026-11-10 | 座位拷到 `hongwei-2026`；在 UB 内扩多 dtype / TF 真假 / 多 rank 泛化 + 云测脚本；**不做**超 UB 分块 | **采纳**（云测 13/13 PASS + 样例 0/120） | EXP_C002 |
| P003 | 2026-11-10 | 性能专题：消灭标量 GetValue/SetValue；优先连续块拷 / Transpose·Gather 原语 / 小 shape 特化；对照平台最优 μs | **进行中**（基线已记） | EXP_C003 + HANDOFF-PERF-PROMPT |
| P004 | 2026-11-10 | A：由实际转置前 shape + perm 合并连续后缀；直接 GM→UB→GM 搬整行，短行在 UB 按 32B 补齐后批量写出；每行仅映射一次源下标，非连续末维保留 P002。假设：连续行路径消除逐元素标量访存，可获得 ≥10× 同机 kernel 加速。**实现/profile 主动用时 ≤90 分钟，首轮云测返回后最多 30 分钟修复；小 shape 累计 ≤2 小时；无收益/证据或超时写 DEC，不转新路** | **候选已实现，待云测，未采纳**；local semantic + 1658 项 CPU 搬运计划通过；平台 shape 未知，不预判覆盖率或豁免 | [EXP_C004](explorations/EXP_C004-contiguous-row-dma.md)；同机 baseline=545f8d2；云测由 Owner 在 CodeArts 执行 |
