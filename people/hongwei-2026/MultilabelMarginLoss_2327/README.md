# hongwei-2026 · MultilabelMarginLoss_2327（初版）

## 来源融合

| 来源 | 吸收内容 |
|------|----------|
| 队友 [yuhongming-2026](../../yuhongming-2026/MultilabelMarginLoss_2327/) baseline（`60f8add`） | InferShape / 三键 / 行分核 / fp32 hinge / bf16 位转换 / `losses[N]` Finalize |
| Codex/Claude `design_final` | `classTile`/`targetTile` UB 滑窗；截断前缀不拆 K；窄写纪律 |
| 本仓 SUMMARY（AI-1/2/3） | 废弃清单：禁 AtomicAdd→y[0]、禁按 K 分核、禁 bf16→half |

## 云端只用这一目录

```text
people/hongwei-2026/MultilabelMarginLoss_2327/workdir/code/
```

**不要**和 `PReluGradUpdate_2329` 混在同一 build 目录；见 `tasks/active/MultilabelMarginLoss_2327/SYNC_CODEARTS.md`。

## 本地

```bash
python people/hongwei-2026/MultilabelMarginLoss_2327/workdir/code/tests/local_semantic_check.py
```

## 相对队友的增量

- TilingData 增加 `classTile` / `targetTile`
- Host：整行 UB 不够时二分 `classTile`（target 仍整行）
- Kernel：按 tile 写 `is_target` + 分块 hinge（大 C 可跑；小 C 退化为单块）
