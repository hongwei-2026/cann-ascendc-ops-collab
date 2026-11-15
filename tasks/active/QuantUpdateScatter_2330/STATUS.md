# 状态 — QuantUpdateScatter_2330

| 项 | 内容 |
|----|------|
| 状态 | **`性能中`** |
| Owner | yuhongming-2026 |
| 更新日期 | 2026-11-15 |
| 现行基线 | **Q002**（未升级；等平台可见点） |
| Q002 平台 | 可见点 **5.33 / 5.74 / 6.52 / 5.67 / 7.83** us |
| Q003 | 废弃（无收益已回退） |
| Q005 | **诊断结案** |
| Q006 | 云 `-e` **PASS**（0/64）；profile narrow/wide_in **COMPLETE**；**平台 NOT_RUN** |
| 工作树代码 | tip `4db2b31` Q006 候选；回退副本 `explorations/Q006/baseline/` |
| 下一闸 | **上传平台**对比 Q002 可见点；补档 `/tmp/qus2330_q006_rows`；&lt;2% → 回退 |

## 一句话

Q006 设备样例已过、诊断 profile COMPLETE；是否优于 Q002 只看平台五个可见点。
