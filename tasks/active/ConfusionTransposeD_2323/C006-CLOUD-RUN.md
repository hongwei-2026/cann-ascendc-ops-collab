# P006步长DMA云测

本次是性能候选，尚无NPU通过或性能收益；P005的UB/跨rank TF/-1限制没有混入修复。只需执行一次：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull --ff-only origin main
if [ -z "${ASCEND_HOME_PATH:-}" ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
DEVICE_ID=0 bash people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/tests/run_c006_cloud.sh
```

脚本重新隔离构建P004=4761239和当前候选，分别样例-e、39项bitwise、7项msprof。耗时主要在两套编译和profile，终端会逐阶段提示；详细构建输出进日志，不刷全量DEBUG。

结束后复制`C006_EVIDENCE_BEGIN`到`C006_EVIDENCE_END`整段回传即可，失败也回传。脚本会直接打印两版功能计数、设备时间、加速比和错误摘要，不需要继续使用C005命令。

完整包另存`/mnt/workspace/exp_logs/ctd2323/`，也自动复制到CodeArts项目文件树：`people/hongwei-2026/ConfusionTransposeD_2323/deliverables/cloud-logs/EXP_C006_*.tar.gz`。若可下载，回传该原包可保留完整证据；无法下载时先粘贴摘要。不能把摘要归档称为完整日志归档。

如终端滚动丢失，按脚本打印的实际run路径重新只读导出：

```bash
python3 people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/tests/p006_report.py /mnt/workspace/problems/ConfusionTransposeD_2323/explorations/EXP_C006_实际时间_随机后缀
```

exit=0仅意味着本轮本地A/B检查完成，不代表平台RE已修复或已达到平台最优。加速只对同机同输入的profile有效，未获得平台case时不宣称平台降至微秒。
