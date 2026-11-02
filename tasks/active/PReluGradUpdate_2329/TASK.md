# 题目：PReluGradUpdate

## 元信息

| 项 | 内容 |
|----|------|
| 题目 ID / problem | 2329 |
| 赛题类型 | 开源仓算子工程（ict） |
| 类型 | A 算子开发 |
| 对标实现 | 昇腾版本内置 PReluGradUpdate 的 **TBE** 实现 |
| 开发语言 | Ascend C |
| 硬件 | 昇腾 NPU（以模板/任务书为准） |
| 模板包 | `materials/`（已解压）；ZIP 原件见 `D:\华为算子比赛\` |
| 平台（模板） | `ascend910b`；输入 grads/features/weights；输出 dx/update；dtype fp16/fp32/bf16 |
| 本阶段 | **多路 AI 分析 → 汇总**；不开发 Kernel |

## 赛题概述

参考内置 PReluGradUpdate 算子的 TBE 实现，在昇腾 NPU 上基于 Ascend C 实现功能一致的算子，完成设计、开发、测试全流程；验收通过后提交至昇腾算子开源仓。

## 验收摘要（抄自赛题，开发前再与 ZIP/官方条文核对）

### 功能

- 与原 TBE 算子核心功能完全对齐  
- 支持原算子对应的所有数据类型、数据格式  
- 比较方式：由**二进制比较**改为**逻辑值比较**  
- **必须实现泛化**；验收用泛化数据  

### 测试

- 参考内置 TBE 算子设计全场景自验证用例  
- 验收：功能、精度、性能全维度（泛化数据）  
- 自验证报告完整、可复现，用例全过  

### 性能

1. 暂仅要求**所有核参与计算**场景下，性能不低于原 TBE 的 **95%**  
2. 小 shape 无法达标（**10us 以下**场景相差 **3us**）：提供性能仿真图与分析结论，证明 Ascend C 与 TBE 一致或优于 TBE  

→ 适用本仓 [性能止损门禁](../../../docs/collaboration/02-perf-gate.md) 的 A/B 分流。

### 精度

- 满足 CANN Judge 平台对应题目精度默认阈值  

## 语义速记（来自模板 reference.py）

```text
dx     = features > 0 ? grads : weights * grads   # weights 按通道广播
update = features > 0 ? 0     : features * grads  # 与 features 同形，不 reduce
```

## 非目标（本阶段）

- 不实现 / 不改 Kernel 业务逻辑（分析未齐）  
- 不串用其他题目结论  
- 不做无台账、无 SUMMARY 的性能预研  

## 下一步（有界）

| 顺序 | 动作 | 时限建议 |
|------|------|----------|
| 1 | 收齐 AI-2 / AI-3，定稿 `analysis/SUMMARY.md` 并回写台账 | 本迭代 |
| 2 | 按 SUMMARY 写设计 `design.md`（最简对齐 TBE） | 设计评审前 |
| 3 | 功能精度基线 → 性能门禁流程 | 见 CONTRIBUTING |

## 参考

- 团队规范：仓库根目录 `CONTRIBUTING.md`  
- Skill：`skills/SKILL.md`
