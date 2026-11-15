# Q003 — 按实际占用分配 UB（2026-11-15）

用户提供 Q002 平台 Pass 截图，展示时间 14:20:36；可见 5 点均下降，隐藏 45 点明细未知。完整本地通过候选保存于本座位 explorations/Q003/baseline/，不会覆盖。

## 修改

- fp32 scales 已直接 DMA 到 fp32 计算区，移除未使用的 raw 分配；bf16 scales raw 按 2B/元素分配。
- 只有存在 zp 才分配 raw+fp32 区；bf16 zp raw 从 4B 改为实际 2B/元素。
- Host 根据量化三元组及可选 zp 核算最坏两槽容量：有 zp 时 26B/元素，无 zp 为 fp16 18B 或 bf16 20B；单行实际占用更低，仍使用保守两槽预算。
- 索引预留以每核 blockFactor、总有效 indices 和 2048 cap 为上界，按索引实际字节数并 32B 对齐。即使满块回退，尾块仍可缓存，不能简单置零预留。
- 保留 8KiB 系统预留、8192 最大 tile、现有双缓冲、同步、精度算术和四签名分发。未修改运行循环或按前 5 点 shape 特判。

## 状态

本地验证进行中；CANN 编译、平台全 50 点复测及新候选 profile 尚未执行。

## 本地验证

- 实际 host 资源预算代码片段由 C++17 编译，13,824 个 shape/dtype/zp/indices/核数/UB 组合及所有启动核的 cache 上界检查通过。
- 独立 Python 预算检查 12,096 组通过；新 tile 不小于 Q002 的旧 tile，UB 总分配（含索引与 8KiB 预留）不越界。
- 同一 CPU API 模型编译实际 Q003 与已保存 Q002 kernel 类，在 32/64/192/256KiB UB 下共 2,432 组与标量结果一致。覆盖 7905/8192/8193/16385 等分块边界、四 dtype 签名、原地/异址、可选 zp、乘/除、空/短/长 indices、2048/2049 cache 边界。
- 原有 512 组固定 tile 内核类模型、1152 组调度对拍、10 项回归、R001 注册/分发/诊断源码契约检查、参考自带探针全部通过。
- 逐字比较确认：本轮仅更改 kernel Init 的内存分配；LoadTile、拷贝、参数加载、量化算术、队列、Process 与 SyncAll 均与 Q002 一致。完整通过版 SHA256 校验一致。
- 构造 32KiB UB/宽513/两待写行/inplace 用例，DMA API 调用由 19 变为 7，体现减少 tile 后的结构性变化；不能换算成平台时延或隐藏点收益。
- 测试使用同步 CPU 桩，不能证明 CANN 编译、设备异步流水正确性或 NPU 性能。完整 stdout/stderr 在 logs/EXP_Q003/。

## 本轮结论

平台 Pass 但可见点相对 Q002 **无收益**（5.51/5.84/6.68/5.46/7.68 vs 5.33/5.74/6.52/5.67/7.83）。按 DEC 已将 `tiling.cpp` + `quant_update_scatter.h` **回退 Q002（796c82e）**。证据：`logs/EXP_Q003/platform-q003-vs-q002.json`、`platform-q003-pass.png`。路线 **废弃**。
