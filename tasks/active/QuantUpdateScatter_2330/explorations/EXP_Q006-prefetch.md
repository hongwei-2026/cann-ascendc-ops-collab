# EXP Q006 — 跨行预取与递增寻址（本地候选）

日期：2026-11-15；起点 main@25dfbaa。角色 Implementer；座位 yuhongming-2026。先读本题 handoff / Q005 medians / skill，登记 Q006 台账与 DEC，再保存 Q002 并改 kernel。用户后续明确仅本地，因此本轮不连接云端、不 commit/push。

## Q005 依据与范围

已核对 copy01/rows01 medians：大异址 var 有整表拷贝税；narrow_rows_in 457.42us、wide_rows_in 40.43us 显示更新循环值得优化。流水利用率不是墙钟分解，没有时间线不能断言 Q002 完全没有重叠。Q006 只改变下一行提交顺序和地址计算，不重开已废弃 Q003，不改 copy、UB/tile 或量化算法。

## 实现

只改生产 workdir/code/op_kernel/quant_update_scatter.h。

1. PrefetchUpdate 只分配输入槽、发起 DMA 并入队；QuantQueuedTile 只消费当前行、量化并输出。
2. 每 batch、每 column tile 预入首行。主循环在 DeQue 当前行前提交下一行输入，两个输入槽实际同时待消费。末行不额外读取，每 batch 排空；middle 之间输入连续，允许跨 middle 预取。
3. src/dst batch 起点各算一次；逐行加 lastDim，middle 结束用 (varAxis-updateAxis)*lastDim 跳步，省去逐行完整乘积。没有新增逐行除法或取模；batchEnd/跳步提至 column 外。
4. 单行 direct 路径保留；对多 batch 单行，仍逐 batch prime/drain，不声称跨 batch 重叠。indices 的 2048 cap 与 fallback 原样，本轮不另改缓存路线。

未改 Init/UB 分配、TQue 深度 2 / InitBuffer 两槽、CopyVarToOut、SyncAll、参数 tile 复用、四签名和入口、RINT 精度链。原始 Q002 保存于本题 seat/explorations/Q006/baseline；source.json 记录 commit 和 hash。

## 具体风险及覆盖

- 队列 prime/steady/drain 与两槽复用：检查队列满/空、最终无 owned tensor；实际观察到两行待消费。
- 尾行/尾列：33/65/129/8193/16385 宽度、多 column、1/2/3 行；DMA/UB guard 和 GM 范围检查。
- middle 跳步：单 update row 跨多个 middle、完整 axis 更新（gap=0）、稀疏更新及多 batch；与独立标量 oracle 比对。
- 短/空/长 indices、2048/2049 cap、copy-only 核、inplace、可选 zp、reciprocal、四 dtype 签名均覆盖。
- 共享 work/int16/half scratch 及输入/输出槽依赖：三个逻辑 engine 的 happens-before 检查覆盖读写顺序，缺 producer 依赖和过早复用的反例应被拒绝。该逻辑模型按队列 API 契约建模，不是硬件执行或 CANN API 实现验证，不能证明实际性能。

## 本地实测

- run_q006_kernel_model.py：3,072 组实际 Q006 / Q002 kernel 类 / 标量 oracle 对拍通过；2,176 组使用队列，896 组 direct/空更新；1,664 组达到两行待消费。
- 四档 UB（32/64/192/256KiB），Q006/Q002 UB 分配及 DMA 数量完全相同；非二次幂 scale、fp32 舍入边界、clip/RINT；源码方法对比确认 Init/copy/quant/index/params 未改。
- test_q001.py：9 项通过，1,152 组参考/调度比较；包含 fp32/fp64 边界与 half 双舍入反例。
- run_r001_contracts.py：实际源码四签名、四 dispatch 与 ACL 错误诊断契约通过。
- 完整日志在本题 tasks/explorations/logs/EXP_Q006/；首轮和最终日志均保留，不删历史。

## 结论与门禁

本地候选已实现，尚未采纳；已采纳基线仍 Q002，工作树 kernel 当前为 Q006 待测候选。CANN 编译、云 -e、设备精度、平台 50 点、narrow/wide 设备复测与时延收益全部 NOT_RUN。CPU 时间不用于速度判断。

**Cursor review**：逻辑与契约通过，已允许 push 候选。注意样例 `-e` 多为单行 direct；平台/多行 profile 才是门禁。无数字前不判收益。失败回退 `people/.../explorations/Q006/baseline/`。
