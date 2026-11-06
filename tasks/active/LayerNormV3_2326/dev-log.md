# Dev log — LayerNormV3_2326

## 2026-11-06

- 做了：按 `LayerNormV3_problem_2326_template.zip` 在 yuhongming 座位补上 InferShape、Tiling、Kernel；reference 改为尾维广播。
- 结果：`python tests/local_semantic_check.py` 全部通过（样例、小 tile、负轴、gamma 短于/长于行、标量 gamma、float16）。
- 卡住：本机没有 Ascend NPU，`build.sh -e` / UT 没跑。
- 下一步：在 910B 上跑 `bash build.sh -e`，看 float 1e-4、half 1e-3。
- 关联道路：P001
