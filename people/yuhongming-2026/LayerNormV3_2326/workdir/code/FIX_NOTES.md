# 相对模板改了什么

模板：`LayerNormV3_problem_2326_template.zip`（题面 2326，LayerNormV3）。

公式与 TBE 一致：在 `begin_norm_axis` 及其后面的维度上做总体方差归一化，

`y = (x - mean) * rstd * gamma + beta`，`rstd = 1/sqrt(var + epsilon)`。

gamma/beta 与 x 的尾维对齐（`begin_params_axis` 起的形状）。IO 为模板注册的 float32 / float16 / bfloat16，ND。归约在 float32 完成，再按 IO 类型回写。比较按题面改为逻辑值（样例 `np.isclose`，float 1e-4、half 1e-3）。

## 改过的模板文件

| 文件 | 改动 |
|------|------|
| `op_kernel/layer_norm_v3.h` | 按行两遍归约 + 仿射；列方向按 UB tile 切；gamma 用 `flat % gammaNum` |
| `op_kernel/layer_norm_v3.cpp` | 增加 bfloat16 入口（schMode=2） |
| `op_kernel/layer_norm_v3_tiling_data.h` | 行/列/gamma/分核/tile/epsilon 位型 |
| `op_kernel/layer_norm_v3_tiling_key.h` | schMode 改为 2 bit：0=fp16，1=fp32，2=bf16 |
| `op_host/layer_norm_v3_tiling.cpp` | 解析轴与 shape，按行分核，16MB workspace |
| `op_host/layer_norm_v3_infershape.cpp` | y 同 x；mean/rstd 从 begin_norm_axis 起为 1 |
| `samples/reference.py` | 去掉错误的右侧 expand，改为尾维广播 + float32 计算 |
| `tests/ut/op_host/test_layer_norm_v3_tiling.cpp` | 期望 tiling 串与三种 dtype 的 key |
| `tests/ut/op_kernel/test_layer_norm_v3.cpp` | 单核 fixture 改成新 TilingData |
| `README.md` | 指向本说明和无 NPU 自测 |

未改：`layer_norm_v3_def.cpp`（dtype/format 已与模板一致）、examples、`samples/case.json`、`samples/sample.py`、UT 公共桩、CMake/build.sh。

新增：`tests/local_semantic_check.py`（无 NPU 的分块/广播自测）。

## 本机自测

`python tests/local_semantic_check.py`：样例、小 tile、负轴、gamma 短于/长于归一化行、标量 gamma、float16 均通过。

## 910B2 样例

2026-11-08，CANN 9.1.0，`DEVICE_ID=0 bash build.sh -j4 -e`：

- 编译成功，三个 schMode 的 kernel `.o` 都生成了。
- 样例 float32：`y` 0/192，`mean` 0/4，`rstd` 0/4，总评 PASSED。

第一次设备结果是 mean 对、rstd 全错。原因是 tile 对齐到 64，有效列只有 48，补零在减均值再平方后变成 `mean^2`，被算进了方差。现已在求和前把这段尾部清成 0，`rstd` 用标量牛顿迭代。

后来在同一块 910B2 上又测了列长不能按 32 字节对齐的形状。DataCopyPad 会把对齐多读进来的元素留在 UB 里，均值因此全错，y 的错误率接近 100%。装载之后同样把 `[valid, tileAlign)` 清零。单 tile 上限收到 1024，5000 列不再触发 ACL 507035。

mean/rstd 的数据类型跟随 gamma。x 为 fp16 或 bf16 时，gamma/beta 可以是同类型，也可以是 float。

同一块卡上的复测：模板样例，列长 7/24/64/100/5000，axis 0，负轴，params 在 norm 前或后，80 行，标量 gamma，fp16，bf16，以及 x 为 half/bf16、gamma 为 float。这些用例的 y、mean、rstd mismatch 都是 0。平台隐藏用例还没有在 Judge 上复跑。
