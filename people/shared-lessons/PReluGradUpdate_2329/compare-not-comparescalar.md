# 优先 Compare(+zero)，慎用 CompareScalar（P022/P023）

样例 WA 恒负支（97/192）时，`CompareScalar` + `Select(count)` 表现像 mask 全 0。

建议：`Duplicate(zero, 0)` → `Compare(mask, x, zero, CMPMODE::GT, count)`，再 `Select(..., count)`。

证据：EXP008（PR#1）· EXP009（hongwei P023）。
