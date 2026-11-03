# Select 优先用 count 重载（P023）

`Select(..., VSEL_TENSOR_TENSOR_MODE, count)` 与 `design_final` / 题包 CANN 一致。

`Select(..., 64, repeats, BinaryRepeatParams)`（P022/PR#1）在平台出现 **Compile Error** 嫌疑；未云测前不要用八参当默认。

比较侧可优先：`Duplicate(zero)` + `Compare`（慎用 `CompareScalar`），见 EXP008/EXP009。
