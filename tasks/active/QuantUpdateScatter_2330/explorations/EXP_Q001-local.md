# EXP_Q001 — 本地恢复与正确性检查（2026-11-15）

## 范围与来源

用户交接指定 QuantUpdateScatter_2330、yuhongming-2026 座位，随后要求“在本地改就行”。本轮只在该座位和本题台账写入，未云测、未 push。协作仓本地快照缺少交接中的本题目录，因此从本地原始解压工程复制源码，排除 build 和缓存；未更改原始解压文件或 ZIP。本记录不替代或覆盖云端历史台账。

## Q001 实现

- CopyVarToOut 按 32B 对齐分核，inplace 同址跳过，保留 scatter 前的 SyncAll。
- Host 核数取 batchCores 与 copyCores 的最大值；每约 32KiB 数据启用一个 copy core，不因 batch=1 再把核数缩回 1。
- 每核 indices 缓存上限 2048 项，超限维持标量回退；int64 最坏缓存额外预留 16KiB UB。
- lastDim tile 外循环加载 scales/zp，再处理本核全部待写行；保留 fp32→int16(RINT)→half→int8。
- 每个 DMA local buffer 按 32B 对齐容量，长度先限制再转 int32；显式保护 vector 数据依赖与更新缓冲复用。
- reference.py 从 fp64 改为逐步 fp32 运算，reciprocal=true 直接乘 scales。
- 修正原 kernel UT：原 fixture 是 MODE_0，却执行 MODE_1，且 tiling 字段未完整初始化；现已对齐。另补 2 个 host 和 4 个 kernel case，尚未在 CANN 编译运行。

## 本地验证

- 9 项 Python CPU 回归全部通过，包含 576 组 tiled schedule 对拍。
- 覆盖 fp16/bf16、int32/int64 indices、原地/异址、可选 zp、乘/除模式、尾块、宽行多 tile、多中间维、短/长/空 indices，以及 2048/2049 缓存边界。
- 独立 scalar fp32 oracle 与参考实现对拍，覆盖除法恰落舍入边界和 fp32→half 二次舍入反例。
- 分核覆盖检查包含超过 int32 范围的大元素数；UB 预算检查包含 32KiB/64KiB/192KiB/256KiB。
- 原 reference 自带探针完成；所有 Python 文件 AST 检查完成；git 变更围栏检查通过。
- 日志为完整 stdout/stderr：logs/EXP_Q001/local-regression.log、reference-probes.log；源码指纹与结构检查另见同目录 JSON。

## 限制与下一闸

以上只是本地参考与调度模型检查，没有执行实际 AscendC 内核。Windows 当前未提供 CANN/NPU 验证，因此 CANN 编译、NPU 精度、同机 TBE/msprof 全部 NOT_RUN，不能记设备 PASS 或任何性能增益。Q002/Q003 未实施。原交接提及的 CLOUD-RUN.md 本地没有副本，本轮没有杜撰云端环境命令。
