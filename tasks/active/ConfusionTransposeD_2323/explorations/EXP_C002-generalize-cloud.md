# EXP_C002 — UB 内扩泛化 / 云测

| 字段 | 内容 |
|------|------|
| 题目 | ConfusionTransposeD_2323 |
| 日期 | 2026-11-10 |
| 作者 / AI | hongwei-2026 · Cursor |
| 阶段 | 功能泛化（实现中） |
| 结论 | **进行中**（CPU 语义先跑；NPU/云测待归档） |
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
| `local_semantic_check.py` | **PASS**（2026-11-10 Windows CPU，8 case + dtype smoke） |
| `npu_regression.py --generate-only` | **OK**（13 case 含可选 bf16） |
| `build.sh -e` | 待云/NPU |
| `npu_regression.py`（真机） | 待云/NPU |
| 日志归档 | 待 → `explorations/logs/EXP_C002/` |

## 下一步

云测 PASS → 更新本 EXP 结论为采纳，STATUS 下一闸可转「超 UB 分块」或性能。
