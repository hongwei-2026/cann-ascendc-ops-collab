# 状态 — SyncBatchNormGatherStats_2332

| 项 | 内容 |
|----|------|
| 状态 | **`实现中`（S008 可交平台；S010 紧凑余差复现待云跑）** |
| Owner / 座位 | yuhongming-2026 |
| 更新日期 | 2026-11-19 |
| tip | Kernel **`140dbdf`**（未改）；诊断 tip **`9e8fd44`** |
| 云测 | S008 写零已消；S010 compact **待云** |
| 平台 | S002–S004 **WA**；**优先交 S008** |
| 提交 | `deliverables/SUBMIT_LINKS_S008.md` |
| 下一闸 | 平台 S008 结果；可选云跑 `CLOUD-S010.md`（勿为 ulp 改核） |

## 一句话

S010 只抽 15 个超差通道做小 C 复现工具，**未改 Kernel**；平台仍交 `140dbdf`。
