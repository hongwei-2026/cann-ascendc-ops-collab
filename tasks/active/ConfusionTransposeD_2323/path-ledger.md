# Path ledger — ConfusionTransposeD_2323

| ID | 日期 | 路径 / 假设 | 结论 | 证据 |
|----|------|-------------|------|------|
| P001 | 2026-11-04 | UB 常驻整输入 + 输出下标反查 src；dtype 按字节宽选 tilingKey 0/1/2/3 | **采纳**（样例 -e PASS 0/120） | EXP_C001 |
| P002 | 2026-11-10 | 座位拷到 `hongwei-2026`；在 UB 内扩多 dtype / TF 真假 / 多 rank 泛化 + 云测脚本；**不做**超 UB 分块 | **采纳**（云测 13/13 PASS + 样例 0/120） | EXP_C002 |
| P003 | 2026-11-10 | 性能专题：消灭标量 GetValue/SetValue；优先连续块拷 / Transpose·Gather 原语 / 小 shape 特化；对照平台最优 μs | **进行中**（基线已记） | EXP_C003 + HANDOFF-PERF-PROMPT |
