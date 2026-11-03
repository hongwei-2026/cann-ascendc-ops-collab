# EXP015 — P027：开 fp16 后仍 bf16 Cast

| 字段 | 内容 |
|------|------|
| 日志 | `explorations/logs/EXP_P027_20261103_131350/` |
| tip | `d10c2a4` |
| 错误 | `not support bf16 type cast` |
| keys | `7d1e7472`（fp16）+ `40c7c86a`（实为 **fp32**，P025 同 hash）双挂 |

## 要点

1. verify 脚本曾误报 `DT_BF16 still active`（`#if` 块内字符串被 grep 到）——已修。  
2. **即使 BF16=0**，只要同 TU 编进 half 标量转换，fp16 **和** fp32 key 都报 bf16 Cast。  
3. 对照：P025 仅 fp32 → 同 hash `40c7c86a` **编过且样例 PASS**。

## 裁决（P028）

默认回到 **仅 fp32**。低精度需另开「独立编译单元 / 原生 half API」道路，不能与 fp32 混在同一 kernel 头里碰 `ToFloat`/`half`。
