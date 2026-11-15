# 状态 — QuantUpdateScatter_2330

| 项 | 内容 |
|----|------|
| 状态 | **`性能中`** |
| Owner | yuhongming-2026 |
| 更新日期 | 2026-11-15 |
| 现行基线 | **Q002**（代码已从 Q003 回退至 `796c82e` 两文件） |
| Q002 平台 | Pass；可见点 **5.33 / 5.74 / 6.52 / 5.67 / 7.83** us |
| Q003 平台 | Pass；可见点 **5.51 / 5.84 / 6.68 / 5.46 / 7.68** us → **无收益，已废弃** |
| Q005 | copy_heavy out/in 29.62/9.10；pipe 证实 MTE 主导、vec≈0；rows01 COMPLETE 待数字 |
| 下一闸 | 导出 rows01 时延+pipe；再立 Q006（copy / 开销 / 真预取 三选一） |

## 一句话

异址=拷贝（MTE2/3）；同址单行更新几乎不算。生产仍 Q002，等 narrow/wide 数字再改代码。

## Q005（2026-11-15）

数字：`explorations/logs/EXP_Q005/copy01-medians.json`；DEC 已更新。云 tgz：`/tmp/qus2330_q005_copy01.tgz`（31M，未入库）。
