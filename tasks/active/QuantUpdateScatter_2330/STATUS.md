# QuantUpdateScatter_2330

- 阶段：性能中（用户交接）；Q001 尚待 CANN 编译、NPU 精度和同机 profile，不能标 PASS。
- 本次目标：只在本地恢复交接所述 Q001，实现多核拷贝、indices 缓存、量化参数复用，并修正参考实现的 fp32 语义。
- 本地仓库：D:/华为算子比赛/ConfusionTransposeD_2323_perf_work，分支 main；当前本地快照未包含交接所述本题代码和台账。
- 座位：people/yuhongming-2026/QuantUpdateScatter_2330/；台账：tasks/active/QuantUpdateScatter_2330/。
- 来源：只读本地 QuantUpdateScatter_2330_unzipped/QuantUpdateScatter_2330/workdir/code/；不复制 build、缓存、ZIP。
- 本次约束：只本地修改；不云测、不推送；保留 fp32→int16(RINT)→half→int8 和拷贝后的 SyncAll。
- 验证边界：CPU 回归仅验证参考语义、分核/UB 边界和调度等价性；不能替代 CANN 编译与设备精度/性能。
- 官方性能 shape、TBE 同机基线、完整历史台账：本地未发现，不能从交接推断测试通过。
- 本轮完成：Q001 本地候选已落盘；9 项 CPU 回归、576 组调度对拍及参考自带探针完成。
- 下一闸：在实际 CANN 环境编译并运行精度，再开展同机 TBE/profile；本轮不进入新性能路线。
