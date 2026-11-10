# EXP_C004 — P004 连续后缀整行 DMA（候选，等待云测）

2026-11-10；hongwei-2026；唯一题目 ConfusionTransposeD_2323；性能中。

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

## 未验证（不得写 PASS）

- 当前主机没有 CANN/NPU；CANN 编译、DataCopyPad/TQueBind 的本版本支持及设备输出均待云端验证。
- build.sh -e、原 13 项 + 新 9 项 bitwise NPU 回归未运行；缺 bf16 时明确 SKIP，不声称 13/13。
- 原版/候选 msprof 尚无结果；未证明 ≥10×，更未证明未知平台 case 已从 1.7ms 降低。
- 平台最优 3.43～3.66us 不是本次已测同机 TBE；不据此使用其他题的 95%/3us 豁免。

## 云端执行 / 日志回收

见 [C004-CLOUD-RUN.md](../C004-CLOUD-RUN.md)。一键脚本使用全新本题目录，顺序构建 baseline/candidate，各跑 build -e、22 项 bitwise 回归、4 组 msprof（短行、较大 half、128 连续元素行、非连续回退）。缺 profile 或解析不到每次调用的一项硬件任务则 exit=3，不将其视为性能通过。

归档含环境、源代码、commit、完整编译/运行日志、逐项结果、原始 profiling、CSV、comparison.json；失败也会生成 tar。Owner 回传后，下载归档到本 EXP 日志目录，回填结果和 ledger；触止损写 DEC，再 push。当前结论：**候选待验，未结案，未采纳**。
