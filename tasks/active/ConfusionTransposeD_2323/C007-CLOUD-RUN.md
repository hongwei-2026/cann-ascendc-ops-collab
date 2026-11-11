# P007 连续行批量 DMA 云测

P007以P004=`4761239`为基线，只比较连续行源起点等差时合并MTE2块DMA的候选。候选提交推送到`origin/main`后，在CodeArts仓库执行：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull --ff-only origin main
if [ -z "${ASCEND_HOME_PATH:-}" ]; then source /home/developer/Ascend/cann-9.1.0/set_env.sh; fi
DEVICE_ID=0 bash people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/tests/run_c007_cloud.sh
```

脚本隔离导出P004与当前候选，分别运行`build.sh -e`、39项逐字节回归和4项msprof（rank2、小shape、128行、较大4D）。profile每项5次warmup+20次计时；成功、输出字节一致且识别到25个Kernel调用才记PROFILED。

退出时会归档完整日志tar到`/mnt/workspace/exp_logs/ctd2323/`及本题个人CodeArts文件树`people/hongwei-2026/ConfusionTransposeD_2323/deliverables/cloud-logs/`。无法下载时，复制`C007_EVIDENCE_BEGIN`至`C007_EVIDENCE_END`回传；粘贴摘要不等同完整tar。

门槛：39/39 PASS、`build.sh -e`通过，128行案例加速≥1.3×，rank2和小shape回退均≤2%。不达标按DEC005止损并回退P004，不开下一路线。平台Runtime Error仍单独记录，不能用本机msprof替代平台证据。
