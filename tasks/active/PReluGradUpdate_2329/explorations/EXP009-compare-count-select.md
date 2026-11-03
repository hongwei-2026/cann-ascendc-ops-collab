# EXP009 — Compare(+zero) + count Select（P023）

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-03 |
| 作者 | hongwei-2026 |
| 关联 | PR#1 / EXP008（P022）· EXP004 · P020 |
| 代码 | `people/hongwei-2026/PReluGradUpdate_2329/workdir/code/` |

## 背景

`yuhongming-2026` PR#1（P022）假设：`CompareScalar` → mask 全 0（WA 97）。改法：`Duplicate(0)` + `Compare`；Select 用 `BinaryRepeatParams` 八参。

平台试交出现 **Compile Error**（未见云端 `-e` 日志）。`design_final` 的 Select 是 **count 单参重载**，八参重载与评测 CANN 不匹配的风险高。

## 裁决（P023）

| 保留 | 丢弃 |
|------|------|
| `Compare(features, zero, GT)` | `Select(..., 64, repeats, BinaryRepeatParams)` |
| 独立 neg/fg；bf16 位转换；SetFlag/WaitFlag | 改他人 `people/yuhongming-*`（协议禁止） |

实现落在 **hongwei** 座位；建议 PR#1 同步改回 count Select 后再云测。

## 待云测 / 平台

```bash
ME=hongwei-2026
# verify PASS 后
rm -rf build && DEVICE_ID=0 bash build.sh -j4 -e
```

期望：至少能编过；样例错数 ≪ 95/97 才算 mask 修好。
