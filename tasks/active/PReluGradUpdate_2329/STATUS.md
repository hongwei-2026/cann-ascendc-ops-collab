# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P016 Codex 单缓冲/UB 已推；mask 仍待修） |
| Owner | 待指定 |
| 更新日期 | 2026-11-02 |
| 基线 | EXP_P015_162823：WA **97/192** ≡ 恒负支 |
| 代码 | `BUFFER_NUM=1`；tiling coef 21/29；usable 80%；Select=`g/wg`,`0/xg` |
| 下一闸 | 云端 pull → 清 build → `-e`；同时继续查 Compare mask |
| 依据 | Codex `design_final` + `explorations/logs/EXP_P015_*` |
