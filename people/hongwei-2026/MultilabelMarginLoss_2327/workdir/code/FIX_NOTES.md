# MultilabelMarginLoss_2327 — hongwei

## EXP_M001
y WA：多核 SetValue/GetValue 与 DMA 冲突 → 改 DataCopyPad + idle SyncAll + usedCores。

## EXP_M002
COMPILE FAIL：`ScanPrefix() const` 内 `tgtBuf_.Get`（TBuf::Get 非 const）。已去掉 method const。
