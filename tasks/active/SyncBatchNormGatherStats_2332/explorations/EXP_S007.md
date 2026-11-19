# EXP_S007 - broad precision matrix

The original ZIP contains one samples/case.json, not the hidden 50-case list.
The existing sample and kernel UT both use C=6, world=1, float32/int32.
S006 passed that sample and its rounding fixture on 910B; this does not prove
that the platform passed.

S007 adds a configurable ACLNN runner and deterministic 157-case matrix.
Single-world, single-count cases compare all four outputs with the existing
reference at the sample tolerance. They exercise four dtype combinations,
rank 1 and 2, aligned and unaligned channel counts, large channels, count
values, attributes, rounding boundaries and non-finite results.

Multi-world and multi-count cases are PROBE only: first/sum alternatives are
printed separately and must be adjudicated against an authoritative source.
All inputs and outputs are saved for reproduction. Exact special-value
comparison is a diagnostic and may differ from platform grading rules.

## Cloud result (2026-11-19)

Host `dfdd97f8…`, tip `9c865d9`:

`TOTAL {"PASS": 144, "FAIL": 1, "ERROR": 0, "PROBE": 12}`

- Oracle cases through C=4097 + dtype/边界/舍入：**PASS**
- PROBE：本次运行的自定义核输出与 **`world_sum` + `count_sum`** 候选一致；未运行独立内置算子，不能证明平台真值
- FAIL：`large_f32_i32_c524289` — `batch_mean` ~半数为 0，`new_mean` 全对 → S008

Logs: `explorations/logs/EXP_S007/`（云上另有全文 `matrix.log` / `results.json`，请 push）。
