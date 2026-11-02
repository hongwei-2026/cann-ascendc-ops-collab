# 开发流水 — PReluGradUpdate_2329

## 2026-03-22

- 做了：建立协作规范仓；登记本题任务卡。
- 结果：规范与分类就绪。
- 关联道路：P001

## 2026-03-22（材料 + 多路分析）

- 做了：从 `D:\华为算子比赛\` 同步并解压 ZIP；通读 Host/Kernel/reference；写入多路分析规范；完成本题 AI-1；拉起 AI-2/AI-3。
- 结果：语义金标准明确；模板 Kernel/Tiling 为空；bf16 key 有风险；update 不 reduce。
- 卡住：当时 SUMMARY 未齐。
- 下一步：当时为收齐 AI-2/3。
- 关联道路：P002、P003

## 2026-03-22（SUMMARY 定稿）

- 做了：AI-2（行片流式）、AI-3（带宽/泛化）落盘；误放根目录的 AI-3 迁入 `analysis/`；汇总裁决切分主路径与废弃清单。
- 结果：`analysis/SUMMARY.md` 已定稿；P003 采纳。
- 下一步：按 SUMMARY 写本题 `design.md`（仍不串题、不提前死磕性能）。
- 关联道路：P003、P004

## 2026-03-22（AI 工作规范）

- 做了：新增 AI 流程规范、AGENTS.md、会话卡片模板、Cursor always-apply 规则；CONTRIBUTING/README 挂接。
- 结果：后续 AI 须按阶段门禁与路径围栏工作。
- 下一步：用会话卡片指挥写 `design.md`（P004）。
- 关联道路：P005

## 2026-11-02（合并本地 Codex/Claude + 云同步）

- 做了：发现 `D:\华为算子比赛\PReluGradUpdate_problem_2329_plan`；导入 design_final；重写 SUMMARY；design.md 对齐；写 SYNC_CODEARTS；建 workdir。
- 结果：实施主依据=本地设计（Select/Inf/模式R/weights[1|C]）。
- 下一步：按 design 改 workdir Host/Kernel → push → CodeArts pull。
- 关联道路：P006、P007
