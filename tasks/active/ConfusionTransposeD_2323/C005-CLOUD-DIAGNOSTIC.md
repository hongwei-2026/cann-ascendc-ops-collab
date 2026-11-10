# P005 / EXP_C005 平台错误诊断

本轮不提交新Kernel、不重编原版/候选Kernel。复用已跑成功的C004隔离构建目录，分别重编一个诊断样例驱动；运行52个边界探测用例并记录失败阶段。

CodeArts终端一次执行：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull --ff-only origin main
if [ -z "${ASCEND_HOME_PATH:-}" ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
DEVICE_ID=0 bash people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/tests/run_c005_diagnostic.sh
```

默认复用 `/mnt/workspace/problems/ConfusionTransposeD_2323/explorations/EXP_C004_20261110_142548_L3wI/`；如果该目录被移动，可在最后一行前指定 `CTD_C004_RUN=/实际C004目录`。若已删除，脚本会明确报缺失，不能靠日志tar代替已编译vendor包。

完成后从 `/mnt/workspace/exp_logs/ctd2323/` 下载并回传本次 `EXP_C005_*.tar.gz`，失败也回传。

注意：输出中的FAIL_OR_REJECTED/TIMEOUT必须保留。exit=0仅表示诊断脚本完成，**不是功能通过，也不是Runtime Error已修复**。诊断只提供本地可复现边界，不会自动证明哪个是隐藏平台case。

## 无法下载时：复制已有Q9AC详情（不重跑）

Q9AC控制台已回收：两版均46/52通过，同6项FAIL_OR_REJECTED；仍缺失败原始错误和阶段计时。在CodeArts执行下列命令，将输出从 `C005_TEXT_EVIDENCE_BEGIN` 到 `C005_TEXT_EVIDENCE_END` 复制回传：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull --ff-only origin main
python3 people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/tests/print_c005_evidence.py /mnt/workspace/problems/ConfusionTransposeD_2323/explorations/EXP_C005_20261110_153111_Q9AC
```

此命令只读取已生成日志，默认candidate失败全文/摘录和代表项host阶段；两版结果分类均保留。每个原始日志打印字节数和SHA256，超过6000字符时明确标为摘录。若需要某项全文，可在最后一行追加 `--version baseline --case ub_188416_bytes --full`（替换相应版本/用例）；不必重新运行诊断。
