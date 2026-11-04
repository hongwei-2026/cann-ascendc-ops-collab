# MultilabelMarginLoss_2327 — hongwei

## EXP_M001
y WA：多核 `SetValue`/`GetValue` 与 DMA 冲突 → 改 `DataCopyPad` + idle `SyncAll` + `usedCores`。

## EXP_M002
COMPILE FAIL：`ScanPrefix() const` 内 `tgtBuf_.Get`（`TBuf::Get` 非 const）。已去掉 method const。

## EXP_M003 / M004 / 平台 1–5
样例与平台用例精度 Pass（M006 tip）。

## M006
无正类早退；全目标 j 块跳过；semantic 扩测。

## M009（候选，云测部分失败）
按需 workspace + 小 reduction 单核。云上 **none / tiny 单核 PASS**；**多核 mean/sum → ACL 507035**（去掉 16MB 系统区后 SyncAll 不够）。

## M010（修复）
多核 mean/sum 恢复 `WS_SYS_SIZE(16MB) + aligned losses[N]`，`rowLoss` 改回 `GetUserWorkspace`。
保留 none / 单核 `workspace=0` 与 tiny 单核路径。
