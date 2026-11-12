# C009 — 探针云测 / 平台提交

本轮候选在 Host infer 与 tiling 写 `CTD_PROBE` JSON，Kernel 与 P004 计算逻辑未改。

**平台限制：不能新增文件。** 探针已内联进既有：

- `op_host/confusion_transpose_d_tiling.cpp`
- `op_host/confusion_transpose_d_infershape.cpp`

**不要**再上传 `confusion_transpose_d_probe.h`（已从座位删除）。

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
GIT_EDITOR=true git -c core.pager=cat pull --ff-only origin main
source /home/developer/Ascend/cann-9.1.0/set_env.sh
DEVICE_ID=0 bash people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/tests/run_c009_cloud.sh
```

平台提交：只交模板既有源码树（含上述两个已改 Host cpp + 原 Kernel）。探针版耗时**不作**性能成绩。提交后回收结果与完整日志；若无 `CTD_PROBE` / 无详细日志，如实回报。
