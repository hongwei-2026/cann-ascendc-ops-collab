# P104 本地候选入口（未云测、未提交、未推送）

更新日期：2026-11-04。基于已测 P103 `47db93e`，仅尝试每核常驻零缓冲；不得把 P103 的 42.89 分作为 P104 成绩。

- 本地代码目录：`D:\sageAttention2 算子开发任务书\people\hongwei-2026\PReluGradUpdate_2329\workdir\code\`
- Kernel：`op_kernel/p_relu_grad_update.h`
- Host 分配预算：`op_host/p_relu_grad_update_tiling.cpp`
- Kernel SHA256（UTF-8 / LF）：`2c5f09e1901a05e81e345ab797d1e62d414cb898acbaa58eda39d60c4e5fcb2a`
- 本地检查：`python -B tests/compute_schedule_check.py` 与 `python -B tests/local_semantic_check.py`
- 云端下一闸：三种 dtype 编译，`bash build.sh -j4 -e`，验证源文件/构建副本指纹一致，保留完整日志；之后平台精度与同环境 P103/P104 对照采时。

注意：本地候选未在 Git 提交中。直接拉取远端仍得到历史源码，不能据此声称已测到 P104；需按用户选择的本地同步方式传递完整当前代码。本轮不自动提交或推送。

变更、内存代价及止损条件见 [EXP024](../explorations/EXP024-persistent-zero-buffer.md)。
