# 优先 Compare(+zero)，慎用 CompareScalar（P022）

样例 WA 恒负支（97/192）时，`CompareScalar` + `Select(count)` 等效 mask 全 0。

建议：

1. `Duplicate(zero, 0)` → `Compare(mask, x, zero, CMPMODE::GT, count)`
2. `Select(..., VSEL_TENSOR_TENSOR_MODE, 64, count/64, {1,1,1,8,8,8})`
3. 仍禁止掩码乘法；保留 Select 独立 src1 buffer

证据：`tasks/active/PReluGradUpdate_2329/explorations/EXP008-compare-repeat-select.md`
