# EXP_C002 — UB 内扩泛化 / 云测

| 字段 | 内容 |
|------|------|
| 题目 | ConfusionTransposeD_2323 |
| 日期 | 2026-11-10 |
| 作者 / AI | hongwei-2026 · Cursor |
| 阶段 | 功能泛化（实现中） |
| 结论 | **采纳**（云测 exit_code=0；样例 PASSED；回归 13/13 PASS） |
| 座位 | `people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/` |
| path-ledger | P002 |

## 范围

- 从 yuhongming 座位 **拷贝** 实现；不改他人 workdir
- UB 内多 case：TF 真/假、rank2/3/4、同 rank reshape、多 dtype
- **不做**超 UB 分块（后置）

## 约束（实现现状）

- tiling 要求 `len(perm) == len(shape)` → 跨 rank reshape 用例不进本闸
- 超 UB 仍 tiling fail（P001 边界）

## 怎么跑

完整命令见 [SYNC_CODEARTS.md](../SYNC_CODEARTS.md)（git push → 云端 pull/rsync → `run_c002_cloud.sh` → IDE Download tar）。

## 结果

| 项 | 状态 |
|----|------|
| `local_semantic_check.py` | **PASS** |
| `build.sh -e` | **PASSED**（y mismatched=0/120） |
| `npu_regression.py` | **13/13 PASS**（含 bf16；最大 larger_4d_f16 8192） |
| 日志 | `explorations/logs/EXP_C002/`（含 `EXP_C002_20261110_134035_YaM4.tar.gz`） |

## 下一步

开超 UB 分块新 path-ledger 行；或先性能基线。
