# 提交清单 — S005 起点（回退 S003/S004 后）

> tip：以 push 后 `main` 最新 commit 为准（本文件写入时尚未 commit）。  
> **不要**再交 `fa21ed5`（S004）包——平台仍 WA 且更慢。

## 七文件（相对座位 `workdir/code/`）

与 S002 相同路径集合；Kernel 已剥掉 Quant/Store 标量保 NaN：

1. `op_host/sync_batch_norm_gather_stats_def.cpp`
2. `op_host/sync_batch_norm_gather_stats.cpp`
3. `op_host/sync_batch_norm_gather_stats_tiling.h`
4. `op_kernel/sync_batch_norm_gather_stats.cpp`
5. `op_kernel/sync_batch_norm_gather_stats.h`
6. `CMakeLists.txt`（若平台要求）
7. 其它平台页所列文件按页为准

## 建议

1. push 本回退后先云 `-e` 确认样例仍绿（应同 S002）。  
2. 再交平台只为确认用时回到 ~5.9μs 量级；**隐藏点仍会 WA**，直到 S005 另修。  
3. 真要修 WA：按 `HANDOFF-S005-PLATFORM-WA-PROMPT.md`，勿重交本回退包当「已修好」。
