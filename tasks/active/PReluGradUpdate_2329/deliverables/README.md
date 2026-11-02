# PReluGradUpdate_2329 提交工程包

| 项 | 内容 |
|----|------|
| 生成时间 | 2026-11-02_201233 |
| 道路 | P022（Compare + 显式 Select repeat） |
| 源码 | `people/yuhongming-2026/PReluGradUpdate_2329/workdir/code/` |
| ZIP（带时间戳） | `deliverables/PReluGradUpdate_2329_code_p022_20261102_201233.zip` |
| 稳定别名 | `deliverables/PReluGradUpdate_2329_code_best.zip` |
| SHA256 | `23FE92C9FBC23498B70E7369F875A33D7E25F78EDD8D866DF1EA93808C9D9E68` |
| 体积 | ~42 KB |
| 结构 | 顶层 `code/`（与官方题包一致） |

## 诚实基线

- 历史云测最好：WA ~95–97/192（mask 失效，见 EXP004/P015）
- 本包相对 hongwei tip 的改动：`Compare(+zero)` 替代 `CompareScalar`；Select 显式 64-elem repeat；bf16 仍位转换
- **尚未** CodeArts `-e` 复测；过样例前不宣称精度 PASS

## 排除项

`__pycache__` / `build` / `*.pyc` / `FIX_NOTES.md`
