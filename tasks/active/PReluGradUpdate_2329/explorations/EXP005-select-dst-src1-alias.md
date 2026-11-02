# EXP005 — Select dst 与 src1 同址导致恒负支全错

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-02 |
| 来源 | Claude 重写实现 + FIX_NOTES；合入 `people/hongwei-2026/.../workdir` |
| 关联 | EXP004（95/97 半错）/ P015 |
| 结论 | **根因更可能是 Select 别名**，不是单纯「Compare 全假」 |

## 现象回顾

seed=42：负支 95 个 → WA 95；或恒负支表现 97。正支大致对、负支全错。

## Claude 定位

`Select` **不保证逐 lane 原子**：若 `dst` 与 `src1` 同 buffer，写 `src0` 时会踩掉尚未读完的 `src1`，mask=0 的 lane 读到被污染的值。  
旧实现里 `xg` 先当 `w*g` 再当 `x*g`，并与 `stashG`/`gF` 复用，易触发该坑。

## 处置

- `negFloatBuf_` / `fgFloatBuf_` 独立存放 `w*g`、`f*g`，不与 Select dst 同址。  
- `CompareScalar` count 按 **64 float（256B）** 对齐；buffer 按对齐后长度申请。  
- `DataCopyPad` 处理尾块。  
- 详见 workdir `FIX_NOTES.md`。

## 待云测

`DEVICE_ID=0 bash build.sh -j4 -e`；bf16 路径仍用 `Cast`，910B 上可能再踩 EXP002（样例为 fp32，先过样例）。
