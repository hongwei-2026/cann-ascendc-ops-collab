# EXP019 — P100 去空 workspace + 全核 channel tile

| 字段 | 内容 |
|------|------|
| 日期 | 2026-11-03 |
| 正确性基线 | P031 平台测试点 1–5 Pass |
| 性能基线 | 1.30 / 1.32 / 2.00 / 1.79 / 1.79 ms |
| 平台参考 | 13.76 / 13.73 / 13.72 / 9.82 / 10.21 μs |

## 瓶颈 1：未使用的 16MB workspace

Kernel 入口明确 `(void)workspace`，但 P026 为错误的 507035 假设把 workspace 固定成 16MB。P031 已证明实际根因是未对齐的 UB `Duplicate`，因此该 workspace 没有正确性用途，只会给每次小算子调用增加固定分配成本。

P100 将 `workspace[0]` 恢复为 0。

## 瓶颈 2：小 tensor 单核 + 串行权重展开

公开 `[4,6,8]` 共 192 个 fp32 元素，旧 tiling 因 `totalNum <= maxTile` 使用 1 核。核内又按 4×6=24 个 channel segment 依次取权重、S_V 同步、构造 weight UB 向量，固定指令开销远大于有效计算。

P100 改为：

- `blockFactor = AlignUp(CeilDiv(M, AIV), 32B元素数)`，公开 fp32 使用 24 核、fp16/bf16 使用 12 核。
- Process 把每个 tile 截断在当前 channel 边界内。
- Compute 每 tile 只加载一个 channel weight，直接执行 `Muls(neg, grads, weight)`。
- 删除 `weightFloatBuf`、per-channel `Duplicate` 和额外的 `Mul`。
- UB 系数从 fp32 29→25 B/elem，低精度 37→33 B/elem。

## 本地验证

- `python tests/local_semantic_check.py`：PASS。
- 覆盖 rank 3/4、`innerSize=1/35/8255`、多核 block 从 channel 中间开始的情况。
- 每个 Compute tile 的首尾 flat index 必须属于同一 channel。
- Host UT 期望更新：fp32 `192 24 8 8 ...`；fp16/bf16 `192 12 16 16 ...`。

## Review 补丁（hongwei）

Codex 原稿 `Process` 在 channel 截断后仍 `progress += tileLen_`，当 `tileLen > channelRemain`（如 `innerSize=1` 或 mid-channel block）会**跳过元素 → WA**。已改为 `progress += currentNum`；Host 在 weights=`[C]` 时把 `tileLen` 上限收到 `innerSize`。

## 门禁

先跑 CodeArts 三 key 编译和 `build.sh -j4 -e`，再交平台采时。若仍为 ms 级，优先检查 aclnn 调用是否重复分配；若降到几十 μs，再单独评估 P101，禁止同时叠加双缓冲和新 dtype 路径。小 shape（&lt;10μs 门禁）按 `02-perf-gate` 分流，勿无限抠。
