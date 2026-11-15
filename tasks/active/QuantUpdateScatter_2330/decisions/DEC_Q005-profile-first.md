# DEC Q005 — 先补瓶颈证据，再改流水

日期：2026-11-15。承接 DEC_Q003，现行代码为 Q002（回退后）。

## 已核实（设备）

`@d8ccf35` 云采 `copy_heavy`（var 8192×256，update 1 行）：

| case | overall median task_us | 说明 |
|------|------------------------|------|
| copy_heavy_out | **29.62** | 异址，需整表拷贝 |
| copy_heavy_in | **9.10** | 同址，跳过拷贝 |

差值约 **20.5 µs**（out/in ≈ 3.3×）。证据：`explorations/logs/EXP_Q005/copy01-medians.json`。

## 瓶颈结论（有界）

- **大 var + 异址**：整表拷贝主导。task Δ≈20.5 µs；pipe 上 out 的 MTE2≈10.85 + MTE3≈9.72，in 的 MTE3≈0 —— 与拷贝税一致。
- **同址 + 单更新行**：vec≈0.019 µs（可忽略）；剩 MTE2≈2.69 + scalar≈2.89。不是「算力不够」，更像参数/调度/标量开销。
- **不能**把 pipe 各通道时间直接相加当墙钟；墙钟仍看 task_duration。
- **不能**仅凭 copy_heavy 断定官方 50 点：inplace/shape 未知。`narrow_rows`/`wide_rows` 已 COMPLETE，数字待入库。

## 决策

1. 生产 host/kernel **暂不改**（仍 Q002）。
2. 待 rows01 数字后立项：
   - 若宽/多行同址仍 vec 极低、scalar/MTE 主导 → 优先减固定开销 / 合并 DMA，而非假双缓冲。
   - 若多行同址出现明显 MTE2↔vec 串行 → 再立真·跨行预取（Q006-pipe）。
   - 仅当平台确认大异址 → Q006-copy。
3. **禁止**重开 Q003；**禁止**无 rows 数字就改生产。
