# MultilabelMarginLoss_2327 — hongwei 初版说明

## P0 已做

1. 语义对齐 `samples/reference.py`（计数 `is_target`、首个 `<0` 截断、`/C` 加权 hinge）
2. InferShape：`none→[N]`，`mean/sum→[1]`，`is_target→[N,C]`
3. 三 tiling key（fp16/fp32/bf16）；bf16 位转换，避免 910B Cast 缺指令
4. 按行分核；mean/sum：`workspace` 存 fp32 `losses[N]`，core0 有序 Finalize
5. **design_final**：`classTile` 滑窗，避免大 C 整行装不下 UB 直接 FAIL

## 刻意未做（P1+）

- design_final 的向量 `Select+ReduceMin` 求前缀 / `accVec` 向量 hinge（需 CPU UT 锁极性）
- 互异正类压缩循环（`D << L_i`）
- none 时 workspace=0 特化（当前仍分配行损失缓冲，实现简单）

## 云测前

CodeArts 只同步本座位 `workdir/code/` → 题目录 `code/`，再 `bash build.sh -j4 -e`。
