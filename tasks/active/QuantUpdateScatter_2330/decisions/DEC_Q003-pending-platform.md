# DEC Q003 — 平台无收益，回退 Q002

日期：2026-11-15。只修改 QuantUpdateScatter_2330 原授权目录。

## 已尝试

- Q003：条件分配 scales/zp UB + host 按 dtype/zp/indices 重算 tile 预算（计算/队列/精度路径未改）。
- 云冒烟 `@ca79bd1`：opc×4 + `mismatched=0/64` PASS。
- 平台全量 Pass；可见 5 点用时见下。

## 可见点（us）

| 点 | Q002 | Q003 | Δ | 榜最优 |
|----|------|------|---|--------|
| 1 | 5.33 | 5.51 | +0.18 | 3.68 |
| 2 | 5.74 | 5.84 | +0.10 | 3.89 |
| 3 | 6.52 | 6.68 | +0.16 | 3.88 |
| 4 | 5.67 | 5.46 | −0.21 | 3.80 |
| 5 | 7.83 | 7.68 | −0.15 | 4.00 |

证据：`explorations/logs/EXP_Q003/platform-q003-vs-q002.json`。

## 瓶颈结论

收紧 UB / 抬 tile **未改变可见点主耗时**（噪声级涨跌）。相对榜最优仍约 1.5～1.9×。瓶颈不在「多省几 KiB UB」，更可能在流水重叠不足 / 拷贝或量化串行（需 profile 后再立新假设）。

## 决策

- **回退** `op_host/quant_update_scatter_tiling.cpp` 与 `op_kernel/quant_update_scatter.h` 至 Q002（`796c82e`）。
- **废弃**「仅靠更大 tile」路线；勿盲开下一路径。
- 下一闸须新假设 + `path-ledger` 新行（建议先 Q005 msprof / 真·跨行双缓冲，或 B 类小 shape 仿真豁免材料）。
