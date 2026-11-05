# PReluGradUpdate_2329 — 个人进展与记录

更新：2026-11-05。当前保留 **J035**；尚未选出可整体替代它的新版本。

| 内容 | 入口 |
|---|---|
| 可审查的完整冻结源码（46 文件） | [workdir/code](../../PReluGradUpdate_2329/workdir/code) |
| 版本指纹、官方与本地结果 | [J035 说明](experiments/J035/README.md) |
| 当前痛点、难点、可证伪方案 | [优化状态](optimization-status.md) |
| 与其他作者实际源码对照 | [源码比较](source-comparison.md) |
| 当前 J053 设备对照 | [实验记录](J053-device-validation.md) |
| 历史路线与停止条件 | [path-ledger.md](path-ledger.md) |
| 本次同步边界 | [会话卡片](session-card.md) |

算子输出为 `dx = x > 0 ? g : w*g`，`update = x > 0 ? 0 : x*g`，两个输出均保持输入 shape；它不做权重梯度归约。通道维为 dim1，权重长度为 1 或 C。

J035 是从本仓框架和既有实现发展出的个人版本，保留原有文件头与来源。`../../PReluGradUpdate_2329/workdir/code/` 内旧 README/FIX_NOTES 为冻结历史内容，当前版本选择和验证范围以本目录说明为准。不要把旧文档的 bf16 后端限制套到当前 910C。

本次公开同步不含大型设备 ZIP、凭据或系统安装文件；关键结果及源码差异已汇总于上述文档，原始数据保留本地。完整设备包的摘要和本地保存状态见 J035 说明。未公开的大包不应被描述成可从 GitHub 下载。

本页是原项目详细入口的保留副本；日常云端同步入口为 [题目 README](../../PReluGradUpdate_2329/README.md)。本目录不作为待同步的工程代码树。
