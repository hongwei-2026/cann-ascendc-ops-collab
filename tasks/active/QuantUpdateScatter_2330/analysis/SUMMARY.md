# Q001 本地执行摘要

本摘要基于用户交接与 2026-11-15 本地检查，不补造历史评审或测试结果。

- 唯一目标：恢复 Q001，完成本地可运行的正确性检查；只改本题指定座位与台账。
- 多核拷贝：按 32B 块分核；tiling 保留 copy-only 核；所有核在 scatter 前 SyncAll。
- 缓存：indices 最多 2048 项，超限标量回退；scales/zp 每 lastDim tile 只加载一次。
- 缓冲安全：DMA 目标按 32B 对齐；UB 另预留 16KiB int64 indices；截断长度后再转 int32。
- 精度：保留 fp32→int16(RINT)→half→int8；reference 修正为 fp32 算术和 reciprocal 直接乘法。
- 已检查到的原始 UT 问题：fixture 实际 MODE_0，原来却执行 MODE_1，并缺失大部分 tiling 字段；本地已对齐。
- 废弃/禁止：直接 fp32→half→int8；删除 SyncAll；把 samples 小 shape 当性能结论。
- 本轮不进入 Q002/Q003；缺少设备基线，不推断优化增益。
- CANN 编译、NPU 精度与同机 TBE/profile 仍待后续验证。
