# DeepSeek 交接：PReluGradUpdate_2329

整理日期：2026-11-04。以下区分本次本地核验、既有记录、控制台显示和待验证方案。本文是交接说明，不代表远程连接或设备测试已恢复。

## 1. 当前委托与停止边界

用户最新要求：先把已有文件、文档和方案整理好，交给 DeepSeek 继续；撤回关闭 TUN 的对照测试。当前只做本地只读整理，不重连，不改 VPN、路由、DNS、代理、安全设置或凭据，不运行设备编译，不向竞赛平台提交。

后续执行者先阅读此文和原始工程记录，再与用户确认下一步执行范围。网络变更须逐项说明具体影响并获得操作时确认。不能把早先“修复网络”的泛化请求当作关闭 TUN 的有效授权。

保留所有已有未提交内容、代码和日志；不删除 VS Code Server，不覆盖其他成员目录，不运行旧安装脚本去重复安装 J001/J002，不自动推送。仓库 AGENTS 中的 push 要求不覆盖用户本次只读整理、不提交的限制。

## 2. 精确位置与版本

正式本地仓库：`E:\suanzi\cann-ascendc-ops-collab`

个人座位：`people/JadeYara/PReluGradUpdate_2329/`。下文的相对路径均相对正式仓库，除非另行标明。

| 对象 | 位置或身份 | 当前状态 |
|---|---|---|
| 本地 Git HEAD | `fc85a48701b1fb578cd8e5208c9b196a471ba267` | 2026-11-04 实读；不是已核最新远端 main |
| J001 源码 | `people/JadeYara/PReluGradUpdate_2329/workdir/code/` | 已在本机，未做本次改动 |
| 固定 P103 基线 | `people/JadeYara/PReluGradUpdate_2329/baseline/P103-code.zip` | 独立归档，勿用旧共享 workdir 代替 |
| P103 来源提交 | `22ec6d63da92ef2193dd66d1666f5d238cf3d08d` | 既有来源记录；P103 构建源记录为 `47db93e0936da5091387ea822a9e4a360f7db6c8` |
| J001 包 | `people/JadeYara/PReluGradUpdate_2329/deliverables/PReluGradUpdate_2329_JadeYara_J001_UNVERIFIED_NPU.zip` | 本次重算 SHA256，42 个包内文件逐一对应 workdir，0 差异 |
| J002 | `people/JadeYara/PReluGradUpdate_2329/experiments/J002/` | 诊断 runner、Python 工具、CPU 记录；不是已修复的提交版本 |
| 本次交接目录 | `C:\Users\lin\Documents\Codex\2026-11-04\task\` | 本文、证据清单与连接采样 |

J001 ZIP SHA256：

`fe319504df5d033fde249f18ca6a091991c3490a21f3bc92ad0d330db6da7775`

Kernel SHA256（既有记录，且本次文件清单提供实际 hash 供核对）：

`2a340f612ec506a6e6ae920fd8f6acd14999f28cb2d89d386d2a075ff6fb3c89`

J001 Host tiling SHA256（既有记录）：

`d561da63c4a280f08a68633a6923a616ced218582d52d226075c1b2d98ef8914`

本次机器可读清单为同目录 `deepseek-handoff-evidence.json`，含源码、个人记录和分析文件的 SHA256、大小、Git 状态及包内文件核验结果。`verify_handoff_sources.py` 只读检查本地源文件并写本目录证据，不联网、不编译。

旧临时制作区在 `C:\Users\lin\Documents\Codex\2026-11-03\task\`，包含 `p2329-stage/`、`j002-stage/`、`j002-p103-cpu/`、`install_j001.py`、`install_j002.py`。它们用于溯源，不能替代正式目录或直接重跑安装。

## 3. 应先读的已有文档

1. `AGENTS.md`、`docs/collaboration/05-ai-work-protocol.md`：协作与写入围栏。
2. `tasks/active/PReluGradUpdate_2329/TASK.md`、`STATUS.md`、`analysis/SUMMARY.md`、`path-ledger.md`、`dev-log.md`：语义、阶段与已废弃路线。
3. `people/JadeYara/PReluGradUpdate_2329/README.md`、`session-card.md`、`SOURCES.md`、`baseline/manifest.json`、`deliverables/manifest.json`：本人候选及来源。
4. `people/JadeYara/PReluGradUpdate_2329/experiments/J002/README.md`：1109131 调查、参数化设备 runner 用法、版本绑定要求。
5. `tasks/active/PReluGradUpdate_2329/analysis/AI-resume-20261103.md`：历史恢复点；以文件顶部的 P103 更新为准，底部保留 P100 旧历史。
6. 同目录 `AI-precision-P103-validation-20261103.md`：16 类设备判别与修复规则；另读 `AI-precision-plan-20261103.md`、`AI-measurement-review-20261103.md`、`AI-numeric-audit-20261103/review.md`。
7. `docs/local-setup.md`：本地配置与交付形态；历史联网状态不是当前网络测量结论。

注意：共享 `STATUS.md` 仍标注 2026-11-02 的 P020/P021，旧 `SUMMARY.md` 还指向共享代码位置；不要据此覆盖 JadeYara 的 J001 或固定 P103。它们与后续个人台账必须合并理解。本次不修改这些历史文档。

本地已有修改包括 `people/CLAIM.md`、`people/JadeYara/README.md` 和本题 `dev-log.md`、`path-ledger.md`；未跟踪项包括个人工程、精度分析、`.editorconfig`、`.gitattributes`、`docs/local-setup.md`。完整快照见证据 JSON。不要执行 reset/clean/整仓覆盖，也不要直接用旧本地台账覆盖更新的远端历史。

## 4. J001 / J002 真正完成了什么

### J001：有限 Host 分派改动

目标是把 `C=1, weights=[1]` 归入共享权重连续 tile 路线，避免按通道模式把 tile 限制在 innerSize。Host 条件从 `weightLen == channelNum` 改为 `weightLen > 1`；合法权重长度为 1 或 C。Kernel、dtype key、转换和搬运路径保持 P103。

本次对基线 ZIP 与 workdir 的只读比较发现已有文件差异仅在：

- `code/op_host/p_relu_grad_update_tiling.cpp`
- `code/tests/ut/op_host/test_p_relu_grad_update_tiling.cpp`

另新增 `tests/shared_weight_check.py`，以交付 manifest 的三项改动记录为准。

既有 CPU 三项检查 PASS；共享权重模型覆盖 816 组。新增四项真实 Host tiling UT 尚未运行。CPU 模型不执行设备指令流水，不能证明 BF16/FP16 设备正确性，也不能证明加速。

### J002：针对 WA 的诊断准备

工具位于 `experiments/J002/`：`device_probe.cpp`、`CMakeLists.txt`、`diagnose.py`、`verify_cpu.py` 及已有日志。已有记录显示：15 项 CPU 故障注入自测 PASS；30 组参考对照 PASS，共 3,609,882 个输出元素；缺失设备输出时检查正确返回失败。

原始证据：`cpu-validation.json`、`local-regression.json`、5 个 P103/J001 CPU `.log`、`wsl-environment.json`。本次只读复核记录，没有重复跑 CPU 测试。

**J002 的 C++ runner 未编译；CANN Host UT、J001/J002 真机精度、同机 P103/J001 性能 A/B 都未执行。没有修复版 J002 提交包。**

### 提交 1109131 尚未绑定版本

既有用户截图报告：2026-11-03 22:53:30，1109131 为 Wrong Answer，点 1 显示错误 100%、3.48 μs，点 2–5 Skipped。没有取得实际上传包及 hash，也没有可定位的失败输入/输出；不能把它当作 J001 的确定失败。3.48 μs 不是有效性能成绩。

后续需要实际上传 ZIP、包内源码身份、编译/运行日志和失败 case。可用 J002 的 `diagnose.py identify` 比较；ZIP hash 不同还须继续比较内部源码，因为压缩元数据也会改变 ZIP hash。

## 5. 已核云端配置与当前连接断点

控制台页面观察到实例 `DevEnvC_YuOAH` 为运行中，标注 ASCEND、1×NPU 910C、40 vCPU、240 GiB；镜像 `cann_9_0_0-py3.11-a3-arm-20261019`；挂载目录 `/mnt/workspace` 和 `/home`。这些是控制台配置，不是 `npu-smi`、CANN 文件或实际编译目标的核验结果。

VS Code 先前打开 `/mnt/workspace`。远端个人工程是否已上传、确切路径、git 版本、CANN 安装路径、910C 编译 SoC 名称均未核实。旧文档部分以 910B/CANN 9.1 为前提，不能原样当作当前 910C/CANN 9.0 的兼容保证。

连接结构：VS Code Remote-SSH → 本机 SSH → `127.0.0.1:31930` AtomGit 转发器 → 云端 WebSocket。插件 `atomgit-cloudenv.cannlabdevspace-1.1.2`；VS Code 1.144.1，Remote-SSH 0.128.0。Host 别名 `devenvc_yuoah.306e9d2caa0846d19bbe3625cf32cdb2.atomgit.0`，账户 `developer`。

以下时间均为北京时间 UTC+8：

| 时间 | 事实 |
|---|---|
| 11-04 00:44:06 | WebSocket 4500，原因 `devenvd disconnected`；随后三次自动重连 HTTP 400，最终 exhausted |
| 16:59:32、17:01:43 | 先前转发进程 `heartbeat inactive`，以 heartbeat-timeout 退出 |
| 17:04:16–17:09:17 | SSH 隧道就绪后，VS Code 通道等待约 300.8 秒才成功 |
| 17:14:40–17:18:42 | 241.95 秒被动观察，49 次原有 TCP 链路采样均保持在线；观察断连/重连 0 次 |
| 同一观察窗 | 48 次本机插件→转发器心跳，间隔 5.003–5.019 秒，平均 5.011 秒；不是云端 RTT |
| 17:33:59、17:35:32 | 后续连接初始化报 `Got bad result from install script`，回到本地欢迎页 |
| 17:36:31 起 | 转发器出现 `Opening handshake has timed out`，后续 10 秒 WebSocket 心跳超时和 `bootstrapForward.reuse.sshBannerUnavailable` |
| 最后一次授权 CLI 探测 | 未得到远端探测输出，退出码 1：`Connection closed by UNKNOWN port 65535`；进程已结束 |

短时正常不能证明长期稳定，也不能覆盖后来失败。SSH 配置中的插件 bootstrap 失败保护会直接退出，足以解释 VS Code 得不到有效安装输出这一现象；没有证据证明 Server 文件损坏，不能据此删 Server。

本机 WinHTTP/系统代理显示关闭，但 `wintun` 为 Up，`0.0.0.0/1` 和 `128.0.0.0/1` 指向 `198.18.0.2`，DNS 也指向该隧道地址。WLAN 网关为 `192.168.3.1`。10 次网关 ping 得到 9 次响应、1 次超时，成功 RTT 1–7 ms、平均 2 ms。样本很小，仅提示本地链路也需核查；尚不能把 VPN、Wi-Fi 或云端任何一项定为唯一根因。

`_handleSocket socket close error` 有 `err=false` 的普通关闭记录，已核对插件日志调用；不能按文案数量当断连次数。工具曾出现 `Transport closed`/执行环境 unavailable，这与云端 SSH 断连也须区分。

原始证据位置：

- 本文同目录 `connection-samples.csv`、`heartbeat-samples.csv`。
- `C:\Users\lin\.atomgitdevenv\.hub\vscode\logs\latest.log`（会继续变化；仅选择性分享，检查敏感信息）。
- `C:\Users\lin\AppData\Roaming\Code\logs\20261104T165848\window1\exthost\output_logging_20261104T170413\1-Remote - SSH.log`。
- 同一 exthost 下 `output_logging_20261104T173254\1-Remote - SSH.log` 与 `output_logging_20261104T173400\1-Remote - SSH.log`。

本文不包含密钥、令牌或带认证查询的 WebSocket URL；不要把整份 SSH/插件配置、身份文件或认证日志发给第三方。

## 6. no_proxy / Docker / Local OJ：待核候选，未执行

下列是用户提供方向的实施前检查框架，不是已验证修复。具体原方案全文未在本地工作线程收到，不能推定未提供的配置和命令。来源链接由父线程核对后转交，本次按用户要求未联网重查。

### no_proxy

`no_proxy`/`NO_PROXY` 仅影响遵循这些变量的客户端。它不会自动绕过 Windows 全局 TUN，也不必然作用于浏览器或 AtomGit WebSocket。

先确定请求由哪个进程、哪个网络库发出，读取其实际代理来源和路由。若之后获得授权，只在单次进程/会话内，对已确认客户端进行最小 no_proxy 对照，记录原值与结果；不要先永久写系统变量，不设全局 `*` 绕过，不更改 TUN/DNS/路由。现有全局隧道路由下，变量变化可能完全无效。

依据：[curl proxy environment variables](https://everything.curl.dev/usingcurl/proxies/env.html)。此来源说明 curl 行为，不能直接当作 AtomGit 插件实现证明。

### Docker / Ascend C CPU 调试

先核对宿主系统架构、驱动、NPU 设备可见性、CANN/镜像兼容及已有 Docker 条件。Windows/WSL 无设备节点的环境即使运行容器，也不会凭空获得 910C。Ascend C CPU 调试可辅助定位逻辑或调试问题，不能证明真机精度、同步行为和性能。

当前旧 WSL 记录未发现 CANN/NPU；不要把 CPU 自测、仿真或容器成功启动写成 NPU PASS。下载镜像、安装依赖、设备映射和权限变更各自先确认所需授权；不使用未经核对的 privileged 或全盘挂载方案。

依据：[Ascend 官方容器概览](https://github.com/Ascend/cann-container-image/blob/main/OVERVIEW.md)。Docker 路线不能单独修复既有 SSH/WebSocket 故障。

### 第三方 Local OJ

候选：[mojohugo/cannjudge-local-oj](https://github.com/mojohugo/cannjudge-local-oj)。这是第三方工具，当前 README 主要面向 910B/CANN 9，需要一台已经可以 SSH 的 Ascend 服务器；不能拿来绕过眼前的连接前置条件。

910C、AArch64、CANN 9.0.0、2329 的接口/包结构及精度比较配置均待核对。先阅读脚本、依赖、上传与运行行为，再决定是否采用；其结果不是官方评分，不代表官方 Judge 通过。安装或执行新第三方软件须按届时适用确认要求处理。

## 7. 恢复后的设备验证顺序（当前不执行）

1. 用户重新授权后才恢复连接排查。分开记录本地网关、TUN/代理路径、AtomGit WebSocket、SSH banner、VS Code Server；当前已明确撤回关闭 TUN，不能自动重试该方案。
2. 连接成功后先低负载持续观察，再读 `npu-smi`、架构、CANN 实际版本、可用编译器与 SoC 目标、内存和当前用户任务，确认可用设备。控制台规格不能替代设备事实。
3. 在个人独立远端目录核实/准备 P103、J001、J002。保存源码与包 hash，核实实际上传 1109131 的包；不覆盖已有工程，不改其他座位，不复用相互污染的 build/OPP 目录。
4. 阅读实际 `build.sh` 与 CMake；先确认 910C/CANN/依赖兼容，然后在独立构建目录编译，保存完整日志。先构建原 P103，再构建 J001。旧文档中的 CANN set_env、GTest/json 路径只是占位符。
5. 执行真实 Host tiling UT、原 FP32 样例和 J002 参数化三 dtype 设备诊断。`build.sh -e` 默认仅 `[4,6,8]`，不能冒充覆盖 J001 改动的 C=1 泛化验证。
6. J002 原用法需结合实际环境核查：CMake 的 `CODE_ROOT` 指向所测源码，`OPP_ROOT` 指向该版本刚构建的包；为 P103/J001 分别使用新 build 和 RUNS 目录。详见 J002 README 的 generate/run 命令，逐条审阅后执行，不把未运行命令当结果。
7. 30 组精确有限值诊断通过后，再跑 `AI-precision-P103-validation-20261103.md` 的 16 类边界/结构清单与原 seed42。保留 expected/actual、dtype/shape、最大误差、失败索引和环境/source/package/runner 身份。
8. 精度通过后才做同机同 shape/dtype 的预热、重复测量、P103/J001 A/B；区分 kernel 时间和端到端时间，报告分布与波动。不存在已证实的 4 μs 承诺或对手实现。

若 P103/J001 同时失败，先查加载包、运行环境和共同实现；若只有 J001 的 C=1 失败，再缩小 case 并查分派/尾 tile。不得先凭猜测重写 Kernel。

## 8. 语义与优化约束

现有 SUMMARY 的语义为：`dx = x>0 ? g : w*g`，`update = x>0 ? 0 : x*g`，通道维 dim1，**update 不做归约**。保持三 dtype 独立正确路径，BF16 不得借 half；不能擅改 golden 或容差。

双缓冲、中间结果复用、同步优化、tiling、分核、msprof 都只是候选。先读实现和测瓶颈；同步依赖必须有真实编译/设备证据，不能因屏障看起来多就删除。原审阅有 6 向量操作/4 屏障及 zero 缓冲先读后写依赖的记录。不要重开已撤回 P102，不改为掩码乘法处理分支，不加 update 归约/atomic。

## 9. 本次交接验收

- 只读核对正式仓库、个人工程、台账与既有日志；未跑新的 CPU/NPU 测试。
- J001 ZIP 42 个文件与 workdir 对应文件 hash 全部一致；候选 ZIP hash 与已有 manifest 相符。
- 只在本次工作目录新增交接文档、校验脚本和证据 JSON；正式工程及原有工作保留。
- 无新网络操作、无 VPN/代理修改、无重连、无安装、无提交推送、无竞赛提交。
- 尚缺：实际 1109131 上传包身份、远程可用连接、910C/CANN 实机核验、Host/CANN 编译与设备精度/性能证据。
