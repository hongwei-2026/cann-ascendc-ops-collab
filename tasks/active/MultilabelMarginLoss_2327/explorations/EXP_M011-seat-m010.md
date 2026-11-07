# EXP_M011 — yuhongming 座位对齐 M010

日期：2026-11-07

## 假设

小 mean/sum 走 1 核、核内 float32 累加，和多核时先写行损失再归约，数值上与 `reference.py` 一致。workspace 只在真正 `SyncAll` 时带 16MB 系统前缀。

## 方法

只改 `people/yuhongming-2026/MultilabelMarginLoss_2327/workdir/code/`：

- `op_kernel/multilabel_margin_loss.h`
- `op_host/multilabel_margin_loss_tiling.cpp`
- `tests/ut/op_host/test_multilabel_margin_loss_tiling.cpp`
- `tests/ut/op_kernel/test_multilabel_margin_loss.cpp`
- `tests/local_semantic_check.py`（补越界标签、空前缀、重复加权、fp16）

## 结果

Python 语义检查 PASS。没有 NPU，未跑 `build.sh -e`。
