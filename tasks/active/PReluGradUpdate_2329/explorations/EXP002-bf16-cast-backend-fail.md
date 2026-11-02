# EXP002 — 910B / CANN 9.1.0：`Cast` bf16 后端不支持

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-02 |
| 环境 | CodeArts · Ascend910B · CANN 9.1.0 · bisheng/clang 15 |
| 关联台账 | P009 / P010 |
| 结论 | **废弃「bf16 路径用 Cast↔fp32」**；改标量/位转换；勿再直接 `Cast(bfloat16_t)` |

## 假设

设计要求 bf16 升 fp32 再 `Select`，实现里对 `bfloat16_t` 调用 `Cast(..., CAST_NONE/RINT)`。

## 现象

```text
fatal error: error in backend: not support bf16 type cast
bisheng: error: clang frontend command failed with exit code 70
ld.lld: ..._2.o: No such file or directory   # kernel_2 = bf16 tiling key，连带 FATBIN 失败
```

`build.sh -j4` / `-e` 均在生成 bf16 binary（schMode/key=2）时炸；fp32/fp16 目标未完整产出。

## 结论（给后人）

1. **910B + 本 CANN 版本不要对 bf16 发 Cast 指令**（至少当前 bisheng backend）。  
2. 缺 `.o` / json / FATBIN 是 Cast 失败的连带症状，不是独立根因。  
3. 正确做法：bf16↔fp32 用 **逐元素位扩展/截断**（或其它非 Cast API）；`Compare/Select/Mul` 仍在 fp32 LocalTensor 上做。  
4. 禁止再试：换 RoundMode 硬编 `Cast(bf16)`、把 bf16 假扮 half。  
