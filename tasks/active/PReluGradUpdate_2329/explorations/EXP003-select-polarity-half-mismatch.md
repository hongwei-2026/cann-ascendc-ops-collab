# EXP003 — 样例 WA ≈ 半错：Select 极性

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-02 |
| 环境 | CodeArts 910B；`-e` 样例 `[4,6,8]` fp32 |
| 关联 | P011 / P012 |
| 结论 | **交换 Select 的 src0/src1**；本机 AscendC 为 bit1→src1（或与设计假设相反） |

## 现象

```text
Build completed successfully!
Wrong Answer: dx.bin, mismatched=97/192
Wrong Answer: update.bin, mismatched=95~97/192
FAILED
```

约一半元素错：与「正/负支对调」高度吻合（随机 features 约半正半负）。

## 假设与处置

原实现按「bit1→src0」：

```text
Select(dx, mask, g, w*g)
Select(upd, mask, 0, x*g)
```

改为「bit1→src1」配对：

```text
Select(dx, mask, w*g, g)
Select(upd, mask, x*g, 0)
```

禁止用 `CMPMODE::LE` 取反（NaN/±0 语义会变）。  
