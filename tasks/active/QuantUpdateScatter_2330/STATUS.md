# 状态 — QuantUpdateScatter_2330

| 项 | 内容 |
|----|------|
| 状态 | **`性能中`** |
| Owner | yuhongming-2026 |
| 更新日期 | 2026-11-15 |
| 现行基线 | **Q002**（代码已从 Q003 回退至 `796c82e` 两文件） |
| Q002 平台 | Pass；可见点 **5.33 / 5.74 / 6.52 / 5.67 / 7.83** us |
| Q003 平台 | Pass；可见点 **5.51 / 5.84 / 6.68 / 5.46 / 7.68** us → **无收益，已废弃** |
| Q005 | `@d8ccf35` 云采 **copy_heavy COMPLETE**（out/in ×3）；数字待归档 |
| 下一闸 | 贴 `results.json` 中位时延 + 必要时 raw CSV；再判拷贝占比 / 立生产路 |

## 一句话

Q002 基线；copy_heavy 设备 profile 已 COMPLETE。先看同址/异址差值，再决定下一条生产优化；禁止盲改 kernel。

## Q005（2026-11-15）

云：`-e` PASS + `profile_q005.sh … copy_heavy_*` → `summary complete: True`。证据目录仍在云 `/tmp/qus2330_q005_copy01`；stdout 已记 `logs/EXP_Q005/cloud-copy01-paste.log`。
