# 路径台账 — QuantUpdateScatter_2330

本文件是 2026-11-15 对当前本地缺失台账的恢复记录，不覆盖或杜撰云端历史。用户要求仅本地修改。

## Q001 — 多核拷贝 / indices 缓存 / scales 与 zp 复用

- 来源：用户交接，已有改动的描述；当前本地原始副本仍为 block0 串行拷贝、逐行加载参数。
- 假设：按 32 字节边界分核拷贝并提高 copy core 数，缓存每核最多 2048 个 indices，按 lastDim tile 复用 scales/zp，可消除重复访存。
- 当前执行：2026-11-15，本地恢复与安全性校核；Implementer：yuhongming-2026。
- 时限：本轮只推进 Q001；未有设备基线不进入 Q002/Q003，不宣称性能增益。
- 允许写：people/yuhongming-2026/QuantUpdateScatter_2330/** 和 tasks/active/QuantUpdateScatter_2330/**。
- 待检查：DMA 尾块 UB 对齐、多核覆盖/同步、int32/int64 索引、缓存超限回退、参考实现的 fp32 运算和 reciprocal 直接乘法。
- 结果：本地候选已完成；9 项 CPU 回归及 576 组调度对拍通过，原参考探针完成。仅此本地检查通过；CANN 编译、NPU 精度、TBE 同机 profile 均 NOT_RUN。完整证据：explorations/EXP_Q001-local.md 与 explorations/logs/EXP_Q001/。

## 后续队列（仅交接记录，本轮不实施）

- Q002：updates 双缓冲，≤1 人日；须先有 Q001 精度与 profile；增益不足 2% 止损记 DEC。
- Q003：UB 缓冲收紧，≤0.5 人日；须有可复现证据再启路。
- Q004：小 shape，≤2h；不以 samples 小用例替代性能门禁。
- Q005：先确认 inplace 与拷贝占比，再决定拷贝专题。