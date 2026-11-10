## 2026-11-10 全历史补录（不是增量进度替代品）

本地 J001–J079 的方向、逐版本尝试、失败和纠偏已整理为 [完整试错历程](full-history/README.md)，分 [J001–J052](full-history/early-trials.md) 与 [J053–J079](full-history/later-trials.md)。[494份原始文本记录索引](full-history/source-index.md)保留来源与SHA256。J041未找到独立结果报告，明确留白；诊断后缀不冒充生产候选。既有简表保留，不再把它当作全部历史。

# 当前入口（2026-11-10）

当前进度截至 J079，见[进度与尝试方向](progress-20261110.md)。工程已更新为官方1221802运行源码加J077复现修复。以下保留历史记录，其中旧 where 语义和 J035 当前基线表述已被新合同及新版入口取代。

---

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
