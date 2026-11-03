# MultilabelMarginLoss_2327 — hongwei（Codex/Claude 修 y WA）

## EXP_M001 根因（已修）

多核 `SetValue`/`GetValue` 写读 `rowLoss`/`y` 走 Scalar DCache，N=4 时同行 64B Cache Line 互踩 → `is_target`(DMA) 对、`y` WA。现全部改 `DataCopyPad`；cnt 从 UB target 重算；空闲核也进 `SyncAll`；Host `usedCores` 按有活核回算。

## 云测

rsync 本座位 code → `problems/MultilabelMarginLoss_2327/code/` → `bash build.sh -j4 -e`
