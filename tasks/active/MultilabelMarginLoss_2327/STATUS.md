# 状态 — MultilabelMarginLoss_2327

| 项 | 内容 |
|----|------|
| 状态 | **`性能中`（入口）**：平台用例 1–5 功能 Pass；耗时远慢于最优 |
| Owner | 共享台账；实现座位见下 |
| 更新日期 | 2026-11-07 |
| 代码座位 | **hongwei M010** 与 **yuhongming 座位** 均为：小形状 1 核、按需 workspace（修多核 507035）。云上尚未用这份座位复测 |
| 队友验证座 | yuhongming：2026-11-04 本地 `-e` PASS；2026-11-07 语义复测 PASS，未重跑 `-e` |
| 设计 | [SUMMARY](analysis/SUMMARY.md) · [design_final](analysis/AI-codex-claude_design_final.md) · [design.md](design.md) |
| 本地闸 | EXP_M003 / EXP_M004：**PASSED** |
| 平台闸 | [用例 1–5 Pass](explorations/PLATFORM_CASES_1-5_20261104.md)（M006；用时 ~713µs vs 最优 ~5µs） |
| 云测 | [EXP_M009](explorations/logs/EXP_M009_20261104_202516_23wV/)：none/tiny PASS；多核 mean/sum **ACL 507035** → M010 修复待复测 |
| 下一闸 | `run_m009_cloud.sh` 复测 M010；再平台对拍耗时 |
| 禁止串题 | 勿与 `PReluGradUpdate_2329` 共用 build/EXP 根目录 |

## 一句话

功能精度已过平台可见点；下一阶段性能（约 150× 差距），先记账再优化。
