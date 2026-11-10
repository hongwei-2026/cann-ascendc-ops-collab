# EXP_C004 — P004 连续后缀整行 DMA（云测通过，采纳实测收益）

2026-11-10；hongwei-2026；唯一题目 ConfusionTransposeD_2323；性能中。

## 云测回传结论（2026-11-10）

- **P004 结案：采纳功能精度及有限性能收益；≥10× 假设未达到，平台目标仍未验收。**
- 环境：910B3，CANN 9.1.0；运行 14:25～14:32（北京时间）；exit_code=0。
- baseline=545f8d2，candidate=4761239；日志内全部 Host/Kernel 源文件与对应 commit 逐个核对一致，云端座位无未提交改动。
- 原版/候选均 build.sh -e PASSED，样例 0/120；均 **22/22 bitwise PASS**，含原 13 项（bf16 已运行）及新 9 项（包括 int64）。
- 原始 op_summary CSV 核验：每版本每 case 25 次 AI_VECTOR_CORE 调用；排除前 5 次，后 20 次任务耗时中位数与 comparison.json 一致。
- 样例 f32（120 元素）：8.50→4.06us，**2.09×**；Block Num 40→3；候选范围 3.90～7.26us，采样存在阶段变化，不能当成所有调用稳定 4us。
- larger_4d_f16（8192 元素）：24.43→10.84us，**2.25×**；40→40 核。
- 连续行 128 元素 f32（16384 元素）：41.07→6.77us，**6.07×**；40→32 核。
- rank2 swap f32（128 元素，span=1 回退）：8.26→8.32us，**0.993×**，耗时上升约 0.73%，没有性能收益；两边均 40 核。
- larger half / 长行的 aiv_scalar_ratio 中位数分别从 0.8165/0.8905 降为 0.504/0.579，支持减少标量负担的方向；DMA 发射和行级映射仍有开销。该比率不作可相加的端到端时间分解。

完整原始包、全文日志、CSV、逐项结果、源码和校验信息已归档：
`logs/EXP_C004/L3wI/`。
原包 SHA256：`14c8db6d052bc36cc734a759258bd991049698cac648f5be9e2dd77c447365d5`。
原始 comparison.json 不改写；补充审计摘要在 `AUDITED-SUMMARY.json`，核数以 CSV 的 `Block Num` 为准（旧解析器漏读该列，已修正）。

## 决策与下一闸

见 [DEC001](../decisions/DEC001-adopt-p004-and-verify-platform.md)：保留已验证 P004，不继续本轮小 shape 调优；先用 4761239 的精确代码重提平台，回收 1～5 点真实耗时。原版同机 8～41us 与平台 1.70～1.82ms 不属已证明的同一测量对象；不能宣称本次快 400×，不能认定平台差距全部来自标量循环，也不能按其他题豁免。

本轮未改 Kernel，只修补性能日志的 `Block Num` 字段识别；后续路线 B 或调度优化须新 ledger 假设与时限。

## 假设与时限（改 Kernel 前已写 P004）

对可合并连续后缀的转置，每行只映射一次 src，以 DMA 代替逐元素 GetValue/SetValue，争取 ≥10× 同机 kernel 加速。实现/profile 主动用时 ≤90 分钟，首轮云测回传后最多 30 分钟修复；小 shape 累计 ≤2 小时。无收益/无证据/到期则写 DEC，不换未经登记的路线。等待 Owner 云端执行不计作实测或 profile 时间。

## 实现及边界

- TF=true 从输入 shape 计算连续 span；TF=false 从 reshape attr shape 计算，正确处理单位维。
- span>1：TQueBind MTE2→MTE3，GM 直接按行搬入；UB 每行补齐到 32B，写出时以 blockCount 批量去掉行尾补齐。按整行与 32B 的公倍数切核。
- span 大于输出 tile 时只切连续行；仍保留原 Host 超 UB tensor 的拒绝边界。
- span=1（例如 rank2 swap）：继续原 P002 标量路径；不宣称覆盖所有未知平台 case。
- dtype key 0/1/2/3 不变，所有宽度只搬字节；新增 copySpan 是真实分派参数。
- 基线 Host/Kernel 精确取 545f8d2。两边共用同一重复执行驱动，每次新建 executor，msprof 排除前 5 次，记录后 20 次硬件任务耗时；CPU 墙钟不作 kernel 性能。

## 已验证

- `local_semantic_check.py`：ALL PASS（CPU）。
- `p004_semantic_check.py`：1658 项 CPU oracle 对 NumPy 精确比较；含 TF/RF、rank1～8、单位维、1/2/4/8B、核边界、UB 短行补齐、4095 行分组上限、长行切片。
- 22 项 NPU fixture 生成完成；**生成不代表 NPU PASS**。
- Python 语法、云测 Bash 语法、git diff --check 通过。
- 本地证据：`explorations/logs/EXP_C004/local/`。

## 首次交付时未验证（历史记录；已由上方回传结论更新）

- 当前主机没有 CANN/NPU；CANN 编译、DataCopyPad/TQueBind 的本版本支持及设备输出均待云端验证。
- build.sh -e、原 13 项 + 新 9 项 bitwise NPU 回归未运行；缺 bf16 时明确 SKIP，不声称 13/13。
- 原版/候选 msprof 尚无结果；未证明 ≥10×，更未证明未知平台 case 已从 1.7ms 降低。
- 平台最优 3.43～3.66us 不是本次已测同机 TBE；不据此使用其他题的 95%/3us 豁免。

## 云端执行 / 日志回收

见 [C004-CLOUD-RUN.md](../C004-CLOUD-RUN.md)。一键脚本使用全新本题目录，顺序构建 baseline/candidate，各跑 build -e、22 项 bitwise 回归、4 组 msprof（短行、较大 half、128 连续元素行、非连续回退）。缺 profile 或解析不到每次调用的一项硬件任务则 exit=3，不将其视为性能通过。

归档含环境、源代码、commit、完整编译/运行日志、逐项结果、原始 profiling、CSV、comparison.json；失败也会生成 tar。首次交付状态为候选待验；本轮回传后已按上方结果结案，平台验收继续待回传。
