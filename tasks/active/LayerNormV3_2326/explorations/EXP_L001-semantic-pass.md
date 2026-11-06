# EXP_L001 — 无 NPU 语义自测

| 项 | 内容 |
|----|------|
| 日期 | 2026-11-06 |
| 道路 | P001 |
| 命令 | `python tests/local_semantic_check.py` |
| 环境 | 本机 Python + numpy，无 CANN、无 NPU |
| 结论 | **采纳**。这只证明分块归约和 gamma 取模与 `reference.py` 一致，不代替 `build.sh -e`。 |

```
PASS sample y(4, 6, 8) mean(4, 1, 1) tile=48
PASS sample-small-tile y(4, 6, 8) mean(4, 1, 1) tile=7
PASS norm2 y(2, 3, 4, 5) mean(2, 3, 1, 1) tile=9
PASS params-after-norm y(2, 3, 4, 5) mean(2, 3, 1, 1) tile=8
PASS params-before-norm y(2, 3, 4, 5) mean(2, 3, 1, 1) tile=11
PASS negative-axis y(2, 3, 4, 5) mean(2, 3, 4, 1) tile=5
PASS whole-tensor y(4, 6, 8) mean(1, 1, 1) tile=13
PASS scalar-affine y(4, 6, 8) mean(4, 1, 1) tile=10
PASS half rtol=1e-3
ALL PASSED
```
