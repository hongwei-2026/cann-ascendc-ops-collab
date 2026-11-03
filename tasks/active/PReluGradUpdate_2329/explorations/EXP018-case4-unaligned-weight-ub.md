# EXP018 — case4 ACL 507035：per-channel weight 的 UB 起点未对齐

| 字段 | 内容 |
|------|------|
| 日期 | 2026-11-03 |
| 现象 | 平台 1–3 Pass；4 Runtime Error `ACL 507035`；5 Skipped |
| 已排除 | 只开 fp32、开三 dtype、half Cast、bf16 位转换/同步均未改变平台症状 |

## 根因假设

旧代码按 channel segment 展开权重：

```cpp
Duplicate(weightLocal[localOffset], weightValue, segment);
```

`weightLocal` 基址对齐，但 `localOffset` 来自 channel 边界。若 `innerSize` 为 1、3、5、35 等非 8-float 倍数，第二个及后续 segment 会从非 32B 对齐的 UB 地址启动向量指令，符合 `VECTOR_CORE_EXCEPTION` 的表现，也解释了为何切换 dtype 无效。

## P031 修复

- 对每个 segment，先用 `SetValue` 写到下一个 8-float 对齐下标。
- `Duplicate` 只从 `fillOffset % 8 == 0` 的位置启动。
- 所有标量前缀写完后执行 S_V 同步，再进行 `Mul`。
- 保留原有对齐 segment 的向量路径，避免全部退化为逐元素填充。

## 本地验证

- `python tests/local_semantic_check.py`：PASS。
- 覆盖 `innerSize=1/3/5/7/8/35/8255`、非零 `globalStart`。
- 静态回归禁止重新出现 `Duplicate(weightLocal[localOffset], ...)`。

## 待验证

CodeArts 清理 build 后编译三 key并执行 `build.sh -j4 -e`；归档完整日志，再重交平台 case4。
