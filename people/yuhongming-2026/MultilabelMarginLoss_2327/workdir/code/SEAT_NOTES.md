# yuhongming-2026 · MultilabelMarginLoss_2327

同步来源：`people/hongwei-2026/.../workdir/code/`（含 EXP_M001 DMA 写回 + EXP_M002 去 ScanPrefix const + classTile 向量 hinge）。

## 本地验证（EXP_M003）

- CANN 9.1.0 · 910B4 · DEVICE_ID=0
- 三 key 编译 SUCCESS
- `PASS: y.bin, mismatched=0/1`
- `PASS: is_target.bin, mismatched=0/24`
- 总评 **PASSED**

## 2026-11-07

对齐多核 `SyncAll` 所需的 workspace 规则：`none` 和实际 1 核时 workspace 为 0，小 mean/sum 强制 1 核并在核内归约；多核才带 16MB 系统前缀。Host/Kernel 单测改成当前 tiling 字段。本机 `local_semantic_check.py` PASS。这次没有 NPU，未重跑 `-e`。
