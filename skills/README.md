# cann-ascendc-op-dev（原仓名 cann-spmv-ascendc）

Cursor Agent Skill：**通用** CANN / AscendC 社区任务算子开发手册（不限 SpMV）。

覆盖：官方认领→设计评审→实现→CI→IT 验收全流程，以及 SIMT/codecheck/交付五件套等实战规范。

## 安装

```bash
git clone https://gitcode.com/hongwei-2026/cann-spmv-ascendc.git ~/.cursor/skills/cann-spmv-ascendc
```

Windows：

```powershell
git clone https://gitcode.com/hongwei-2026/cann-spmv-ascendc.git "$env:USERPROFILE\.cursor\skills\cann-spmv-ascendc"
```

Skill 标识名：`cann-ascendc-op-dev`（见 `SKILL.md` frontmatter）。

## 文件

| 文件 | 说明 |
|------|------|
| `SKILL.md` | 主 skill（通用流程与规范） |
| `official-requirements.md` | 官方 README/模板/讨论要求摘录 |
| `pitfalls.md` | 通用踩坑表（含 `$ASC` 污染 / Fixpipe UB） |
| `acceptance-pack.md` | 通用验收打包清单 |
| `examples-basic-api-cube.md` | Basic API CUBE 指针化实战（asc-devkit） |
| `examples-spmv.md` | SpMV 实战附录（示例） |

## 官方入口

- https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md  
- https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md  
- https://gitcode.com/org/cann/discussions/39  

## License

文档以学习复用为目的；请同时遵守 CANN 开源仓与社区任务许可。
