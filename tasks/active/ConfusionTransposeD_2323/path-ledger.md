# Path ledger — ConfusionTransposeD_2323

| ID | 日期 | 路径 / 假设 | 结论 | 证据 |
|----|------|-------------|------|------|
| P001 | 2026-11-04 | UB 常驻整输入 + 输出下标反查 src；dtype 按字节宽选 tilingKey 0/1/2/3 | **采纳**（样例 -e PASS 0/120） | EXP_C001 |
| P002 | 2026-11-10 | 座位拷到 `hongwei-2026`；在 UB 内扩多 dtype / TF 真假 / 多 rank 泛化 + 云测脚本；**不做**超 UB 分块 | **采纳**（云测 13/13 PASS + 样例 0/120） | EXP_C002 |
| P003 | 2026-11-10 | 性能专题：消灭标量 GetValue/SetValue；优先连续块拷 / Transpose·Gather 原语 / 小 shape 特化；对照平台最优 μs | **进行中**（基线已记） | EXP_C003 + HANDOFF-PERF-PROMPT |
| P004 | 2026-11-10 | A：由实际转置前 shape + perm 合并连续后缀；直接 GM→UB→GM 搬整行，短行在 UB 按 32B 补齐后批量写出；每行仅映射一次源下标，非连续末维保留 P002。假设：连续行路径消除逐元素标量访存，可获得 ≥10× 同机 kernel 加速。**实现/profile 主动用时 ≤90 分钟，首轮云测返回后最多 30 分钟修复；小 shape 累计 ≤2 小时；无收益/证据或超时写 DEC，不转新路** | **结案：采纳有限收益**；910B3/CANN9.1.0 样例PASS+22/22 bitwise；连续行2.09/2.25/6.07×，rank2回退0.993×；**≥10×未达，平台待验**；冻结Kernel | [EXP_C004](explorations/EXP_C004-contiguous-row-dma.md) + [DEC001](decisions/DEC001-adopt-p004-and-verify-platform.md)；完整 L3wI 云日志已归档；candidate=4761239 |
| P005 | 2026-11-10 | **仅诊断，不换优化Kernel**：平台重提1/2点约717/728us，3～5仍1.76～1.84ms且总状态Runtime Error；假设是未覆盖布局/输入边界或平台计时范围差异，尚无根因。复用C004原版/候选的精确Host/Kernel，扩测短tensor、UB边界、跨rank、reshape -1；另记host API墙钟，绝不与C004设备时间混算；抓取本机910B Gather/Transpose头文件供后续决策。**准备≤30分钟，首轮回传诊断≤30分钟；本轮不改Kernel、不做超UB分块，证据不足写DEC结案** | **52用例诊断脚本已备，待云端执行**；仅CPU生成/语义/语法验证；截图未显示失败点/shape/错误详情，不得认定点6或某条代码为根因 | [EXP_C005](explorations/EXP_C005-platform-re-and-boundary-diagnostic.md)；[DEC002](decisions/DEC002-platform-runtime-error-diagnostic-first.md)；Owner确认平台不提供详细错误 |
