# yuhongming-2026

- GitHub：https://github.com/yuhongming-2026
- 座位：2（已认领）

## 本题

- `LayerNormV3_2326/workdir/`：按模板实现两遍 LayerNorm。2026-11-08 在 910B2 上样例 `-e` PASSED。说明见 `tasks/active/LayerNormV3_2326/`
- `MultilabelMarginLoss_2327/workdir/`：计数截断 + hinge。2026-11-07 对齐小形状单核和按需 workspace。语义自测 PASS，这次未跑 `-e`
- `PadV3Grad_2328/workdir/`：按 reference 把 Pad 梯度折回。语义自测 PASS，未跑 `-e`
- `PReluGradUpdate_2329/workdir/`：自 hongwei 基线复制后做 **P022** mask 通路优化
  - `Compare` + 零向量（替代 `CompareScalar`）
  - `Select` 显式 64-elem repeat
- CodeArts：只 rsync **本目录** `workdir/code/`（见题内 `SYNC_CODEARTS.md`）

## 开工

1. `git pull`
2. 读 `../shared-lessons/` 与 `../../tasks/active/PReluGradUpdate_2329/`
3. 探索写在本题 `tasks/.../explorations/` 或本目录 `notes/`
