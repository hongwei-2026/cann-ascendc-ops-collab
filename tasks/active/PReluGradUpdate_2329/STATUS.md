# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P015 已改 Kernel；待云端 `-e`） |
| Owner | 待指定 |
| 更新日期 | 2026-11-02 |
| 基线 | EXP20261102_161116：95/192 ≡ 恒正支（EXP004） |
| 代码 | CompareScalar + Barrier；Select 恢复 `g/wg` 与 `0/xg` |
| 下一闸 | 云端 pull → 清 build → `bash build.sh -j4 -e`；打包日志含 bins |
| 依据 | `explorations/EXP004-mask-always-src1.md` |
