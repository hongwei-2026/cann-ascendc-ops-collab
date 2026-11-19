# 状态 — SyncBatchNormGatherStats_2332

| 项 | 内容 |
|----|------|
| 状态 | **`实现中`（S010 紧凑用例已复现参考差异，待独立裁决）** |
| Owner / 座位 | yuhongming-2026 |
| 更新日期 | 2026-11-19 |
| tip | Kernel **`140dbdf`**（未改）；诊断 tip **`9e8fd44`** |
| 云测 | S008 写零已消；S010 12 次均 invstd 7/15、new_variance 8/15 超差 |
| 平台 | S002–S004 **WA**；**优先交 S008** |
| 提交 | `deliverables/SUBMIT_LINKS_S008.md` |
| 下一闸 | 确认系统设备分发并设计 reference/系统双对照实验；获取 S008 平台结果 |

## 一句话

S010 系统接口对照与 reference、自定义结果均有差异；尚不能断定平台隐藏点根因，Kernel 未改。
