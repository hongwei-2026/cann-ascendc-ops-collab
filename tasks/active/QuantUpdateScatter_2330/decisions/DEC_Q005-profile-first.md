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

- **大 var + 异址**：整表 `CopyVarToOut` 是主耗时来源之一（同机对照成立）。
- **同址路径**：已接近可见点量级（~9 µs），剩余差距更可能在量化行路径 / 固定开销 / 流水，而非「再省 UB」。
- **不能**仅凭本诊断断定官方 50 点主因：平台 case 的 inplace / shape 未知；可见点 5–8 µs 可能本就偏同址或更小 tensor。

## 决策

1. 生产 host/kernel **暂不改**（仍 Q002）。
2. 下一条生产路必须二选一立项（先台账）：
   - **Q006-copy**：仅当确认平台大量异址大 var → 加速整表拷贝（更大 DMA / 更好分核）。
   - **Q006-pipe**：若平台以同址或小拷贝为主 → 真·跨行双缓冲 / 减 Sync，先用 narrow_rows + wide_rows profile。
3. 下一步测量：导出 copy_heavy 的 `aiv_mte2/mte3/vec_time` 中位，并采 `narrow_rows` / `wide_rows` 同址异址对。
4. **禁止**重开 Q003；**禁止**无新 profile 就改生产。
