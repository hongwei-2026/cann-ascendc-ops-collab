# 状态 — PReluGradUpdate

| 项 | 内容 |
|----|------|
| 状态 | `实现中`（P0 已推 workdir；待云端 `-e` 复测） |
| Owner | 待指定 |
| 更新日期 | 2026-11-02 |
| 基线 | EXP001：空 Kernel `Wrong Answer`（dx 192/192） |
| 代码 | `workdir/code` 已按 design_final 填 InferShape/Tiling/Kernel |
| 下一闸 | 云端 pull → build → `DEVICE_ID=0 bash build.sh -j4 -e` |
| 若仍 WA | 优先查 Select 极性（交换 src0/src1），记 EXP002 |
