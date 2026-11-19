# 状态 — SyncBatchNormGatherStats_2332

| 项 | 内容 |
|----|------|
| 状态 | **`实现中`（S004 废弃；代码已回退 S002 Quant/Store；开 S005）** |
| Owner / 座位 | yuhongming-2026 |
| 更新日期 | 2026-11-19 |
| tip | **`52b16fb`**（S005 起点 = S002 Quant/Store） |
| 云测 | EXP_S004 曾 PASS；回退后需再 `-e` |
| 平台 | S002/S003/S004 均为 **WA**（1–5 Pass；隐藏失败） |
| 提交 | 勿再交 S004；见 `HANDOFF-S005-PLATFORM-WA-PROMPT.md` |
| 下一闸 | S005：换假说修隐藏 WA；**禁止**重开 Cast 后全 tile 标量保 NaN |

## 一句话

S004 平台仍 WA，且可见点更慢（~6.1–6.7μs vs S002 ~5.9–6.2）；已剥掉 S003/S004 标量环，下一路查 count/W/dtype。
