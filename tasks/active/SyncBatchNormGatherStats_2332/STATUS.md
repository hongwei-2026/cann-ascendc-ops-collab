# 状态 — SyncBatchNormGatherStats_2332

| 项 | 内容 |
|----|------|
| 状态 | **`实现中`（S008 云测写零改善，S009 分析余差）** |
| Owner / 座位 | yuhongming-2026 |
| 更新日期 | 2026-11-19 |
| tip | Kernel **`140dbdf`**（S008：`MTE3_V`+`MTE3_MTE2`） |
| 云测 | S008 严格矩阵 `PASS=144 FAIL=1 ERROR=0 PROBE=12`；大 C batch_mean 三次均 0 错 |
| 平台 | S002–S004 **WA**；S008 尚未取得平台结果 |
| 提交 | `deliverables/SUBMIT_LINKS_S008.md` |
| 下一闸 | 平台验证 S008，或以独立内置算子对照剩余 7 / 13 处误差 |

## 一句话

S008 三次消除了大 C 的 batch_mean 写零，仍有 7 / 13 处超差；按仓库样例比例规则低于 0.01%，但平台规则尚未核实。
