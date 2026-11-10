# Q9AC — 仅控制台粘贴证据

Owner回传的原始附件全文保存为 `console-paste.log`，未删改提示符/换行；字节数7093、SHA256 `bd877ba7ff55c6d52af50e2a4b621542c06c2c29d2ea3bb31245c3d9b126835a`。`console-summary.json`由104条逐case输出解析，非手工拼接PASS。

原run：`/mnt/workspace/problems/ConfusionTransposeD_2323/explorations/EXP_C005_20261110_153111_Q9AC`。控制台最后显示tar已生成，**本机未取得该tar**，也未取得逐case原始错误、host阶段、源码核验manifest或64份API头文件。

两版均46项PASS_BITWISE、6项FAIL_OR_REJECTED且逐项分类相同；不是52/52验收通过，不说明平台RE已修复。下一步按 `C005-CLOUD-DIAGNOSTIC.md`只读导出既有详情。

## 第二次文本回传

`detail-paste.log`原文52713字节，SHA256 `42d2ffbbd42e16cc2b7ef54d3bd016f99f4c56e6af371add1b4a3343887900d0`；解析见 `detail-summary.json`。原诊断commit为9b6de1c；12条失败记录均exit=2、最后为GetWorkspaceSize开始标记，无launch标记。取得candidate四项主机阶段计时及六份失败日志的远端字节数/哈希，未取得完整原文件。

每份失败日志约4311万字节，第一次导出仅首尾摘录，错误正文不在摘录中。后续使用 `--errors` 扫描全文筛错误及上下文；不要求粘贴43MB全文，也不重跑设备。主机计时来自调试日志全开的冷启动进程，不是kernel性能结果。
