# EXP_S007 — 云矩阵结果

| 项 | 内容 |
|----|------|
| tip | `9c865d9`（Kernel 仍为 `b2271b4`） |
| 主机 | `dfdd97f8c4e84acbaa54982714e4d993` |
| self-test | **PASS**（157） |
| `build.sh -j8 -e` | **PASS**（含 `s007_runner`） |
| matrix | **TOTAL `PASS=144 FAIL=1 ERROR=0 PROBE=12`** |

## FAIL（1）

- `large_f32_i32_c524289`
- `batch_mean`：**249859/524289** 错（样例 index 起多为 `actual=0.0`）
- `batch_invstd`：7 处 ulp 级
- `new_mean`：**0** 错
- `new_variance`：13 处 ulp 级  
→ 同 tile 算出了正确 `new_mean`，但 `batch_mean` 大片为 0：优先查大 C 写回 / `SetGlobalBuffer` 长度 / 多核覆盖（S008）。

## PROBE（12）

设备输出与候选比对：**稳定对齐 `world_sum` + `count_sum`**（`world_sum_count_sum` 四输出 mismatched=0）。  
`world_first` / `count_first` 组合系统偏离。  
→ Kernel 现状=「设备维求和 + count 数组求和」；**不要**仅凭 reference 的 `[0]` 去改成取首元素。平台真值仍须下载工程/内置裁决。

## 结论

单设备单 count（含 4 dtype、尾块、rank1、舍入、特殊值、C≤4097）对齐 reference。  
隐藏 WA 下一优先：**大 C（S008）**；语义探针勿当 FAIL。
