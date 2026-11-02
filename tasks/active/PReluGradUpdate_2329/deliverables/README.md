# PReluGradUpdate_2329 提交工程包

| 项 | 内容 |
|----|------|
| 生成时间 | 2026-11-02_181641 |
| git tip | `beaa00e`（P020：bf16 位转换 + Select 独立 buffer + SetFlag/WaitFlag） |
| 源码 | `people/hongwei-2026/PReluGradUpdate_2329/workdir/code/` |
| ZIP（带时间戳） | `deliverables/PReluGradUpdate_2329_code_best_20261102_181641.zip` |
| 稳定别名 | `deliverables/PReluGradUpdate_2329_code_best.zip` |
| 仓库根副本 | `PReluGradUpdate_2329_code_best.zip` |
| 本机桌面 | `C:\Users\yuhon\Desktop\PReluGradUpdate_2329_code_best.zip` |
| SHA256 | `A7C800F60130748B1315E6E71D34ACF6FF069CE011C92891703D31620E7F973D` |
| 体积 | ~41 KB / 39 entries |
| 结构 | 顶层 `code/`（与官方题包一致） |

## 诚实基线

- 云上曾编过的最好样例：WA ~95–97/192（`EXP_P015` / `EXP_P016` / `EXP20261102_161116`）
- tip `beaa00e` 含 bf16 去 Cast，**尚未** CodeArts 复测通过；P019 因 bf16 Cast 炸 FATBIN
- 本包适合先交一版试平台；不保证全量精度 PASS

## 排除项

`__pycache__` / `build` / `*.pyc` / `FIX_NOTES.md`
