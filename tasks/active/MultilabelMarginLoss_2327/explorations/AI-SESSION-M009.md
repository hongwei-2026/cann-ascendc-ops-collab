# M009 session

题目：MultilabelMarginLoss_2327。用户已确认优化本地 hongwei_submit.zip。

基线：协作仓 b61feb7，hongwei M006；ZIP 与仓内 Host/Kernel 的 LF SHA256 一致。
Kernel：b9adc43409fbea44e6eb5dbb795f90b160450d3e1a8d277a671c62058007d89b。

角色：Implementer，基于已认领的 hongwei-2026 基线制作隔离候选。
本地候选在 D:/华为算子比赛/MultilabelMarginLoss_2327_M009_work；协作仓尚未写入。
允许路径：people/hongwei-2026/MultilabelMarginLoss_2327/ 与
tasks/active/MultilabelMarginLoss_2327/。不改其他座位或 2329，不推送。

唯一目标：按需申请归约 workspace，移除无消费者的 16MB 固定预留；
微小 reduction 在单核有序完成，避免 workspace 往返与 SyncAll。
损失计算、计数语义、DMA 写回及多核有序 Finalize 保持同一算法。

时限：本次有界实现与本地检查；真机前不宣称提速或功能通过。
退出条件：输出可审查 diff、候选 ZIP、源码指纹、验证记录和隔离云测命令。

本机只有 Windows/Python/MinGW，没有 CANN 编译器或 NPU；CPU 检查不能代替云测。
