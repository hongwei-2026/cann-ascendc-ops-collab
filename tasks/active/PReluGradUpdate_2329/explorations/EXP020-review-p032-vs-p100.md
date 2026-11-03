# EXP020 — 审 yuhongming P032 vs hongwei P100

| 字段 | 内容 |
|------|------|
| 日期 | 2026-11-03 |
| 队友 tip | `3e3358d`（P032） |
| Owner tip | `3bd65fb`（P100） |

## P032 结论

- **采纳（座位正确性）**：weight `Duplicate` 对齐与 EXP018/P031 一致。  
- **部分重叠**：tiling `WS_SYS_SIZE=0` 与 P100 一致。  
- **勿作默认试交**：仍保留八参 Select、weight UB 展开、小图单核；缺 P100 的 channel-`Muls` 与 `progress+=currentNum`。

## 分工

功能基线 = P031；性能默认包 = P100。队友若做性能实验，在自有 `people/` 吸收 P100 全量后再开新道路。
