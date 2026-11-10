# P004 / EXP_C004 一键云测

仅题目 ConfusionTransposeD_2323、hongwei-2026 座位。候选未验证，先跑下面命令；脚本在本题全新目录分别构建原版和候选版。

CodeArts 终端：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull --ff-only origin main
if [ -z "${ASCEND_HOME_PATH:-}" ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
DEVICE_ID=0 bash people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/tests/run_c004_cloud.sh
```

输出末尾会显示 `C004 exit=...; download: ...tar.gz`。从 `/mnt/workspace/exp_logs/ctd2323/` 下载对应 `EXP_C004_*.tar.gz` 并回传，失败也要保留完整包。

- exit=0：两个版本的样例、回归、硬件时间采集成功；仍需阅读 comparison.json 判断优化是否达假设。
- exit=3：功能流程走完，性能采集缺失/异常；不能判定性能通过。
- 其他非零：见包内完整编译/回归日志，不代表通过。

22 项包含原有 13 项与新增 9 项；依赖未安装导致 bf16 缺失会明确 SKIP。所有已运行项用字节精确比较。4 个 profile 用例固定 shape，不能代替隐藏平台 case；本轮未自动提交平台。

归档位置：`tasks/active/ConfusionTransposeD_2323/explorations/logs/EXP_C004/`；云日志回收并 push 后才可结案。无需覆盖旧运行区或删除旧日志。
