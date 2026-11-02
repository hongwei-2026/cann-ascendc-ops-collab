# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（Select 极性已推；最新云测仍 WA） |
| Owner | 待指定 |
| 更新日期 | 2026-11-02 |
| 基线 | EXP001：空 Kernel WA；EXP20261102_161116：Build OK，dx/update 各 95/192 |
| 代码 | `workdir/code` 含 InferShape/Tiling/Compare-Select（极性已换） |
| 下一闸 | 云端确认已 pull f89dd20 → 清 build → `bash build.sh -j4 -e` |
| 若仍 WA | 查权重广播 / Compare 语义；日志见 `explorations/logs/EXP20261102_161116/` |
