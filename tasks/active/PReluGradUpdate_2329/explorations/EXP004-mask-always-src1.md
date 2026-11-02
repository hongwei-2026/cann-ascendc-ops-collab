# EXP004 — 半错实为「mask 失效 / 恒选 src1」

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-02 |
| 环境 | CodeArts 910B；样例 `[4,6,8]` fp32，`sample.py` seed=42 |
| 日志 | `explorations/logs/EXP20261102_161116/build_e_full.log` |
| 关联 | P012（Select 极性假设）→ **废弃为根因**；本 EXP 为新根因 |
| 结论 | **Compare 掩码未生效，Select 恒走 src1**；EXP003 换极性只把「恒负支」换成「恒正支」 |

## 1. 日志事实

```text
Build completed successfully!
Wrong Answer: dx.bin, mismatched=95/192
Wrong Answer: update.bin, mismatched=95/192
FAILED
```

`bins/` 空（云端未拷到 output/golden），只能对照 `reference.py` + 同 seed 复现计数。

## 2. seed=42 复算（本机 numpy）

| 量 | 值 |
|----|-----|
| `x>0` | **97** |
| `x<=0` | **95** |
| 恒输出 `dx=g`（正支） | dx 错 **95** |
| 恒输出 `update=0`（正支） | update 错 **95** |
| 恒输出 `dx=w*g`（负支） | dx 错 **97** |
| 恒输出 `update=x*g`（负支） | update 错 **97** |
| 极性完全反转 | dx/update 各错 **192**（与日志不符） |

当前云测 **95/95** ≡ 核上结果是 **`dx≡g` 且 `update≡0`**（恒正支）。

## 3. 与 EXP003 的衔接

| 阶段 | Select 写法（现行注释） | 若 mask 全 0（恒 src1） | 预期错数 |
|------|-------------------------|-------------------------|----------|
| EXP003 前 | `Select(dx,mask,g,wg)` / `Select(upd,mask,0,xg)` | dx=`wg`, upd=`xg` | ≈**97** / ≈**97** |
| EXP003 后 | `Select(dx,mask,wg,g)` / `Select(upd,mask,xg,0)` | dx=`g`, upd=`0` | **95** / **95** |

与两次云测一致：**换 src0/src1 只换了「恒选哪一侧」**，说明 **mask 全程未区分正负**，不是「极性反了还会半对」的模型。

真极性反转会接近 **192/192**，本日志不是。

## 4. 根因候选（按优先级）

1. **`Compare` 结果未进入 Select 所用 mask**（布局/API：bitpack vs 逐字节、`Compare` vs `CompareScalar`、缺 `PipeBarrier`）。
2. **`xF` 侧比较无效**（比较对象/同步问题导致 GT 恒假 → mask 全 0 → 恒 src1）。
3. 权重广播等 **次要**：单独错通道映射约为 73，对不上 95。

## 5. 处置（下一实现闸，未改代码前）

1. **撤回**「为修半错而交换 Select」；恢复 design 语义：`dx=mask?g:w*g`，`upd=mask?0:x*g`（在 **bit1→src0** 约定下即 `Select(..., g, wg)` / `Select(..., zeros, xg)`）。
2. 修 mask 通路：优先试 `CompareScalar(xF, static_cast<float>(0), CMPMODE::GT, calcLen)` + Compare/Select 间 `PipeBarrier<PIPE_V>()`；核对 mask 字节预算（bit 掩码按 `Ceil(count/8)` 对齐）。
3. 用 B03 四点（`x=[1,-1,0,NaN]`）或至少再跑 `-e`：若仍 95→应接近 0；若变 97→仍在恒 src1。
4. 下次打包 **必须**拷 `samples/build-npu/output|golden|input`，禁止空 `bins/`。

## 6. 废弃

| ID | 内容 |
|----|------|
| P012 作为根因 | 「半错 = Select 极性反」——计数证明为 mask 失效 |
