# MultilabelMarginLoss_2327 — hongwei

## EXP_M001
y WA：多核 `SetValue`/`GetValue` 与 DMA 冲突 → 改 `DataCopyPad` + idle `SyncAll` + `usedCores`。

## EXP_M002
COMPILE FAIL：`ScanPrefix() const` 内 `tgtBuf_.Get`（`TBuf::Get` 非 const）。已去掉 method const。

## EXP_M003（队友验证）
yuhongming 同步本 tip → 本地 910B4 样例 `-e` **PASSED**（y 0/1 · is_target 0/24）。

## M006（hongwei 跟进）
- 吸收 M003 结论；座位仍为云同步源之一。
- Kernel：无正类早退；全目标类 j 块跳过内层 O(C)。
- 扩 `local_semantic_check`：K≠C、任意负截断、空正类、classTile mirror ≡ reference。

## M009 candidate (not yet tested on NPU)
- Workspace on demand: none/single-core=0; multi-core mean/sum=aligned losses[N].
- Tiny mean/sum use one core; preserve per-row /C and final /N order.
- Row-loss GM pointer uses the dedicated raw workspace; no system GM consumers.
- Host/kernel UT fixtures updated; optional example arguments and 72-case NPU runner added.
- See M009_README.md. Default validated baseline remains M006 until cloud checks pass.
