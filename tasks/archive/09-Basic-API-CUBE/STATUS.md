# 状态 — Basic API CUBE 指针化（归档）

| 项 | 内容 |
|----|------|
| 状态 | `归档` / 未最终验收通过 |
| 合入仓 | cann/asc-devkit |
| 设计 | competitions !1594 |
| 代码 | fork hongwei-2026/asc-devkit · 上游 MR !6109 |
| 环境 | hidevlab Ascend950PR / CANN 9.1.0 / dav-3510 |

## 高优先级教训（禁止再踩）

- `$ASC` **只 sync 改动文件**，禁止 tip 整棵 `dav_3510/` / `impl/utils`  
- Device-only 指针重载门控；clang-format 18.1.8；`stat/needs-squash` → 单 commit  
- 详见 `skills/pitfalls.md` §E、`skills/examples-basic-api-cube.md`
