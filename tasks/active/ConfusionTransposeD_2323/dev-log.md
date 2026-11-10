# Dev log — ConfusionTransposeD_2323

## 2026-11-10

- 授权：拷贝座位到 `people/hongwei-2026/ConfusionTransposeD_2323/`，先做扩泛化/云测（不改 yuhongming workdir）
- path-ledger **P002** 开工；加 `tests/local_semantic_check.py` / `npu_regression.py` / `run_c002_cloud.sh` + 参数化 example
- 本机无 Ascend：先跑 CPU 语义；写 `SYNC_CODEARTS.md`（git 同步 + exp_logs/ctd2323 下载归档）
- NPU/云测待执行后归档 `explorations/logs/EXP_C002/`

## 2026-11-04

- 从 `ConfusionTransposeD_problem_2323_template.zip` 建座位与台账
- 实现 infershape / tiling / kernel（输出下标 → 输入下标）
- 本地 910B4 `bash build.sh -e`：**PASSED**（y mismatched=0/120）
