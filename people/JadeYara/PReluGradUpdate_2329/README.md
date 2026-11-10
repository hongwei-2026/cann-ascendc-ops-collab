# JadeYara · PReluGradUpdate_2329

更新：2026-11-10，进度截至 J079。

- **完整本地试错历史（J001–J079）**：[方向总览、逐版本结果和原始文档](../notes/PReluGradUpdate_2329/full-history/README.md)。这是全程复盘，下面的进度页仅是当前状态。

- **云端同步工程**：[workdir/code](workdir/code/)，当前是 **官方 1221802 运行源码 + J077 复现/UT 修复**，不是 J078 失败候选。
- **当前进度、尝试方向、成绩与下一闸**：[进度总结](../notes/PReluGradUpdate_2329/progress-20261110.md)。
- **个人台账**：[path-ledger](../notes/PReluGradUpdate_2329/path-ledger.md)；[实验记录](../notes/PReluGradUpdate_2329/experiments/)。
- **同步源码指纹**：[49 文件 SHA256](../notes/PReluGradUpdate_2329/workdir-source-manifest-20261110.json)。
- **历史 J035**：[冻结副本](../notes/PReluGradUpdate_2329/experiments/J035/frozen-code/)，原记录与 Git 历史保留。

官方 1221802 五点为 7.25 / 7.33 / 7.44 / 5.67 / 5.73 μs。J077 没有改运行 Host/Kernel；其构建/测试修复不等于新的官方成绩。J076 云测整体节省中位仅 0.0475 μs，J078 不采纳，J079 尚未证明单包计时替代方案通过。

只同步自己的 workdir/code；不要覆盖其他作者目录。正式评测由用户提交。本次只整理并发布既有结果，没有新设备实验。
