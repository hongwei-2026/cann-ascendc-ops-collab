# C009 — 探针云测

本轮候选在Host infer和tiling写`CTD_PROBE` JSON，Kernel和P004计算逻辑未改。探针包含新头文件`op_host/confusion_transpose_d_probe.h`，平台若需要手工拷源码，必须连同两个Host `.cpp`和该头文件一起提交。不要在云端验证前提交平台。

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull --ff-only origin main
source /home/developer/Ascend/cann-9.1.0/set_env.sh
DEVICE_ID=0 bash people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/tests/run_c009_cloud.sh
```

脚本导出P004和当前提交到独立目录，分别执行`build.sh -e`和39项bitwise回归；候选另外运行C005边界case，检查tiling JSON的输入、属性和输出存储shape，以及UB拒绝阶段。未观察到infer退出会列入`infer_unobserved`，不代表infer通过；infer若出现但不匹配则失败。输出`C009_EVIDENCE_BEGIN`至`C009_EVIDENCE_END`，请完整回传；原始日志tar位于输出的`archive=`路径，并复制到`/mnt/workspace/exp_logs/ctd2323/`。若下载不可用，回传marker并说明原始包无法下载。

脚本失败时同样回传marker。若日志不包含`CTD_PROBE`，先确认Host库构建与日志捕获，不得把缺失结果记为通过。`dtype_code`是GE枚举；`host_ok`只描述infer或tiling回调。回调可能缓存、并行或根本未进入，不能按序号直接映射平台点。stderr是否在平台可见未知；平台无日志时不可能从本地模拟推断隐藏shape。
