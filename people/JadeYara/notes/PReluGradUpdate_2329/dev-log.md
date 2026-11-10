# JadeYara 开发日志 — PReluGradUpdate_2329

## 2026-11-05 — 接手与个人目录同步

- 核对两份用户附件、本地交接、J035/J053 源码哈希和 J050–J052 归档；独立 clone main `e62693c8bd5946eebcf2ffaacb37d2bcb892bbbf`，保留原本地未提交工作。
- 按个人目录模板同步 J035 完整 46 文件至 `workdir/code`，记录官方结果来源、当前测量难点、其他作者源码差异和可证伪方案；未改变冻结源码内容。
- 使用既有 SSH 转发恢复连接，确认 CANN 9.3.0 / compiler、OPP 9.3.0-beta.1，驱动 25.5.5，NPU 无运行进程后启动 J053。
- J053 首次测试入口配置找不到头文件；实际文件在 CPack staging，改用该私有目录后 runner 编译成功，135 个函数字节核对均一致。保留完整失败和修复日志；后续结果见 [本轮设备记录](J053-device-validation.md)。
- 用户已明确授权认领，CLAIM 第 3 行改为“已认领”、日期 2026-11-05，其他席位保持不变。GitHub 连接账号为 JadeYara，该仓库返回 push 权限。官方提交仍由用户操作。

- J053 本轮各 780 精度、5184 压力通过；648 窗口/226800 kernel 记录完整，162 组控制均通过，最大 A/A 差 0.059 μs。目标低精度约快 0.44 μs、高复用约快 1 μs；本轮完成，不再追加重复测试或生产分派。

## 2026-11-05 — 按 hongwei 实际目录整理

- 直接核对 `people/hongwei-2026/README.md`、题目 `README.md`、完整文件树、`workdir/code/README.md`、`FIX_NOTES.md` 与 `verify_codearts_source.sh`，并同步最新 main；未以模板代替实际对照。
- JadeYara 账号入口改为账号/席位/当前题目；题目入口采用朋友的“云端版本 → 共享台账 → 共识教训”组织，填写自己的 J035、结果和当前难点。
- 工程仍为 `PReluGradUpdate_2329/workdir/code/`；已有详细文档、台账和实验说明移动到 `notes/PReluGradUpdate_2329/` 保留，链接随目录更新。
- 朋友的云端核对脚本包含其 P030/CANN 9.1 标记，不将这些规则冒充 J035 的验证，也未复制成新的测试。本次无源码、性能或精度测试变更。

| 朋友实际位置 / 用途 | JadeYara 对应位置 |
|---|---|
| `people/hongwei-2026/README.md`：账号、席位、当前题目 | `people/JadeYara/README.md` |
| `PReluGradUpdate_2329/README.md`：同步版本、共享资料入口 | `PReluGradUpdate_2329/README.md`，内容为 J035 |
| `PReluGradUpdate_2329/workdir/code/`：Host/Kernel、工程 README、FIX_NOTES | 同层级，保留 J035 完整 46 文件 |
| `verify_codearts_source.sh`：朋友已有的 P030 同步检查 | 不照搬其版本判断；本次仅整理文档，不新增脚本或测试 |
| 朋友在共享题目录记录 EXP；个人入口链接过去 | 同样链接共享记录，自己的原有文档另外保留在个人 `notes/PReluGradUpdate_2329/`，未移入他人或共享目录 |

## 2026-11-10 同步截至 J079
- 核对本机 J077–J079 与 J076 交接证据，编写最新进度和可证伪方向。
- workdir/code 同步官方1221802运行实现及 J077 构建/UT 修复；保留旧 J035 完整副本。
- 公开关键原始文本日志、结果与失败；不上传大 ZIP、二进制或不完整包。
- 下一闸：标准名单包独立进程空改控制；未新增 NPU 实验或官方提交。
