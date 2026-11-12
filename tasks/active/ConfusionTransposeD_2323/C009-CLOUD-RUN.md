# C009 — 探针云测 / 平台合规

## 平台合规（强制）

平台拒绝：**新增文件**、**Debug 探针**（`CTD_PROBE` / `fprintf(stderr)` 等）。提示原文大意：「提交代码存在不合规内容…如需 Debug 请在本地进行。」

因此：

- **平台提交**：只用干净 P004 Host/Kernel（当前 `op_host/confusion_transpose_d_{tiling,infershape}.cpp` 已无探针）。
- **探针**：仅 CodeArts / 本地（`tests/c009_*`、云测脚本）；**不要**再往平台塞探针。

## CodeArts 探针云测（可选）

若需在仓内保留可编译探针实验，另开分支或临时改 Host，**勿 push 到准备提平台的工作树**。历史探针云测结果见 `explorations/EXP_C009-platform-shape-probe.md` 与 `explorations/logs/EXP_C009/`。

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
GIT_EDITOR=true git -c core.pager=cat pull --ff-only origin main
source /home/developer/Ascend/cann-9.1.0/set_env.sh
# 当前 main Host 已无 CTD_PROBE；run_c009 若依赖探针需用带探针的历史提交或本地补丁
```
