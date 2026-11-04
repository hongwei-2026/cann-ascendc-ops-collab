# EXP_M009: workspace on demand

状态：候选实现及本地检查完成；云测/采时待做，默认通过版本仍为 M006。

## 证据与假设

平台 1/2/3/5 约 713–715us，case4 2.10ms；最优约 5us。
样例 N=4,C=6,K=6。基线不区分 reduction/核数，始终申请 16MB+4N；
none 不使用它，单核 reduction 也不需要跨核行损失缓冲。
这些源码事实支持先验证固定开销；日志没有 profiler，不能断言全部时间来自 workspace。

## 实现边界

- none：保留按行分核，workspace=0。
- mean/sum：只有单核时在寄存器按行累加；最后 mean 仍单独除 N。
- 微小 reduction：N<=8,C<=32,K<=32 且 N*C*K<=4096 时单核。
- 其他 mean/sum：保留按行分核、DMA losses[N]、硬件 SyncAll、core0 有序 Finalize。
  workspace 仅申请向上对齐到 32B 的 N 个 float，不预留 16MB。
- 本算子没有使用系统 GM workspace 的 Matmul/HCCL/软同步 API。
  losses 从传入 workspace 首址开始，避免 GetUserWorkspace 的系统区偏移假设。
- 不修改 hinge、重复计数权重、负数截断、dtype 键、输出窄 DMA 写回。

## 验证与下一闸

- 现有 `tests/local_semantic_check.py`：PASS。
- `checks/check_m009_control_flow.py`：提取实际 C++ `Process()` / `SetWorkspace()`，
  使用 MinGW C++17 编译受检查的控制流替身：36 个归约流程、42 个 workspace 场景 PASS。
  包括 workspace=null 的 none/单核、N=100/64 核的空闲核参与同步、N>512 分段 Finalize、全局 `/N`。
  这不是 CANN CPU twin，不模拟 DMA 异步/核间可见性，也不验证 CANN 编译器。
- 三 dtype × 三 reduction × 八组 shape：72 组独立 reference 输入/金标已生成。
  其中覆盖 K≠C、重复计数、任意负截断、空/全目标、单/多核、小任务边界、C 滑窗、N>512。
  本机未执行其 NPU 端；不能记作 72 组精度 PASS。
- Host UT 更新了 workspace/核数/三 key 的期望；原 Kernel UT 的过期 totalNum/blockFactor/ubFactor
  修为实际 tiling 字段并传 null workspace。SDK UT 未运行。
- Python 语法及 Git diff 空白检查通过；原逐行数值路径无改动。

本地日志在 `logs/EXP_M009_local/`。
Kernel LF SHA256：`a0d66d3bf2fa38f157b23e390a69fe5670a9345f05c294450d08d9701914dd0e`。
Host LF SHA256：`fe00ab9f526660830cf66ef59134d6fb2c875ec675ad1955e497fe43f5558481`。

运行目录中的 `DEVICE_ID=0 bash tests/run_m009_cloud.sh` 会校验指纹、构建、运行原样例
和 72 组回归，并将日志/源码/输入/金标/实际输出打包。也可仅运行 `bash build.sh -j4 -e`。

CANN 9.x 编译、三 dtype 真机精度及平台逐点耗时均待验证。
先特别确认零 workspace、原 case4 和多核 workspace 边界；再同环境比较 M006/M009。
若功能失败或无收益，使用保留的原始 M006 ZIP 回退，不能以候选替代已通过结论。
