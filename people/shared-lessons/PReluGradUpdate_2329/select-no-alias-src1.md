# Select：dst 不要与 src1 同 buffer

AscendC `Select` 可能非逐 lane 原子。`dst` 与 `src1` 别名时，mask=0 的 lane 会读到已被 src0 覆盖的值 → 常表现为**正支对、负支全错**（错数≈非正元素个数）。

做法：`w*g`、`x*g` 各用独立 TBuf；Select 的 dst 另开。  
见 EXP005 / workdir `FIX_NOTES.md`。
