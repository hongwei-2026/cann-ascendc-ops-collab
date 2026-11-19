# 状态 — SyncBatchNormGatherStats_2332

| 项 | 内容 |
|----|------|
| 状态 | **`实现中`（S007 云矩阵出结果；开 S008 修大 C）** |
| Owner / 座位 | yuhongming-2026 |
| 更新日期 | 2026-11-19 |
| tip | Kernel **`b2271b4`**；测试 tip **`9c865d9`** |
| 云测 | EXP_S007：`PASS=144 FAIL=1 ERROR=0 PROBE=12` |
| 平台 | S002–S004 **WA**；S006 可交但大 C 未修前风险高 |
| 提交 | 建议先修 S008 再交；清单仍见 `SUBMIT_LINKS_S006.md` |
| 下一闸 | S008：修 `large_f32_i32_c524289` 的 `batch_mean` 大片 0 |

## 一句话

矩阵证明小/中 shape 与 reference 对齐，且 Kernel 语义是 world/count **求和**；唯一 FAIL 是 C=524289 的 `batch_mean` 写零，优先修它。
