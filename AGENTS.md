# AGENTS.md — AI 进入本仓必读

你是本仓库的 AI 协作者。详细规则：`docs/collaboration/05-ai-work-protocol.md`。  
违反围栏或跳阶段 = 失败，即使代码「看起来能跑」。

## 启动检查（每次会话）

1. 用户是否给出 **题名**（如 `PReluGradUpdate_2329`）？没有 → **只读**，先问清。  
2. 读 `tasks/.../<题>/STATUS.md`：当前阶段允许做什么？  
3. 读该题 `path-ledger.md` +（若有）`analysis/SUMMARY.md` 的 **废弃清单**。  
4. 本会话 **唯一目标** 是否一句话可复述？多目标 → 让用户拆开。  
5. 填写心理/书面围栏：只改哪些路径？

## 硬禁止

- 串题、改 `tasks` 下其他题目、在仓库根乱丢分析/代码  
- 无 `SUMMARY` 定稿就深度编码 / 性能专题  
- 无 `path-ledger` 新行就换优化路线  
- 假 PASS、删历史台账/分析、扩大用户未授权 scope  
- 多 Implementer 同时改同一题代码  
- 把子 Agent 文件随便写在题根；必须进 `analysis/` 等约定目录

## 阶段速查

| STATUS | 你可以做 | 不要做 |
|--------|----------|--------|
| 分析中 | `analysis/AI-*.md` | 写 Kernel |
| 分析定稿 | `design.md` | 无确认的大实现 |
| 实现中 | SUMMARY 的 P0 | 跳去 P2 死磕 |
| 性能中 | 有界优化 + DEC | 小 shape 无限抠 |

## 结束前

更新本题 `path-ledger` + `dev-log`；汇报改动路径与下一闸。  
完整流程与失控恢复见 `docs/collaboration/05-ai-work-protocol.md`。
