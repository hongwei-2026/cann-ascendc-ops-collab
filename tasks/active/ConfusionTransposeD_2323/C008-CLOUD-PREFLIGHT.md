# C008 — CodeArts TBE 与 API 能力盘点

本闸只读检查 CodeArts 的 910B / CANN 9.1 环境，不构建候选、不修改 Kernel，也不把平台 leaderboard 的 best time 当作 TBE 基线。

## 运行

先把本机刚推送的 `main` 拉到 CodeArts 协作仓，再运行：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
GIT_EDITOR=true git pull --ff-only origin main
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/tests/run_c008_preflight.sh
```

如果 `set_env.sh` 路径不同，改用 CodeArts 当前 CANN 9.1 环境的真实路径。脚本会检查芯片和 CANN 版本、TBE 文件/注册符号、`Transpose`/`Gather` 头文件声明；不访问平台私有数据，也不猜隐藏点 shape。

## 回传

从终端复制完整的 `C008_EVIDENCE_BEGIN` 到 `C008_EVIDENCE_END`。完整日志包会保存到 `/mnt/workspace/exp_logs/ctd2323/`。若可以下载，把 `EXP_C008_*.tar.gz` 放入 `tasks/active/ConfusionTransposeD_2323/explorations/logs/EXP_C008/` 后提交；若仍不能下载，回传 marker 原文并标成粘贴证据。

## 结案条件

- 找到可在同一 910B / CANN 9.1 设备调用的 TBE 实现，并可获得与 P004 相同输入上的 kernel task 时间：进入下一条有界 Kernel 假设前，补齐 P004/TBE 对照。
- 找不到 TBE 实现、不能绕开自定义 op 注册冲突，或缺少可映射的目标输入：在 60 分钟预算内止损并写 DEC；不改 Kernel，不声称平台最优就是 TBE。
