# PReluGradUpdate_2329 接手调研报告（本阶段不做优化）

- 编制：2026-11-07，Asia/Shanghai
- 范围：接手项目资料、复核证据链、昇腾社区与跨平台调研、云端连接实测、提出优化方向与最小判别实验
- 边界：**未修改任何算子代码、未运行 CANN 构建/精度/性能/设备实验、未登录官方平台、未提交或推送、未修改 SSH/网络/系统配置**
- 结论性质：本报告区分「已证实」「仅相关」「未验证」；凡"官方成绩"均只按用户历史确认记录引用，本次未独立核验

---

## 0 速览

| 问题 | 结论 |
|---|---|
| 项目接手 | 完成。最新工程 `R=C:\Users\lin\Documents\Codex\2026-11-05\task-2\publish-repo`（HEAD `a68973e`，195 项未提交/未跟踪，**不可 reset/clean**）；旧副本 `E:\suanzi\cann-ascendc-ops-collab`（HEAD `fc85a48`，97 项），只作历史参考。生产基线仍是 **J035**；J056 有用户确认的官方 Pass；J059 是有效独立候选；J060/J061 不采纳；**J062 停在归档断点、不晋升整包**。 |
| 云端连接 | **部分连通，不能记为"成功"**。本次从本机到转发层：`127.0.0.1:31930` 在监听、TCP 15 ms 可连；但读不到 SSH banner、`ssh-keyscan` 无输出，`ssh <alias>` 被 AtomGit `Match … !exec bootstrap` 门禁改走 `ProxyCommand cmd.exe /d /c exit 1`（退出码 255）。**未到达主机身份/认证/远端执行层**。探测前约 6 分钟 VS Code 仍成功连过一次（09:40:48–09:40:54）。**可信主机指纹核验未完成**（本次未取得 offered 指纹）。 |
| J062 BF16 回退 | **原始战役中真实存在（A/C 一致、B 单独偏），但 2026-11-07 云端 E0 重跑无法复现**。19 组控制失败应拆三类：**12 组＝未改路径 BF16 S=1 系统性退化**（`[127,64,1]` +18.5%~19.0%、`[127,8192,1]` +5.0%~5.2%，六轮恒定）、2 组＝BF16 S=33 临界越界（+2.07%~2.26%）、5 组＝长 FP16 输入的 A/A 稳定性越界（B 实际 6/6 胜，不是回退）。关键签名：AIV **scalar 时间 +1.15~1.20 μs、vec 时间 0 变化**，block_nums 与五张量地址在 A/B/C 间完全一致。源码级可排除"BF16 S=1 换实现"；回退落在该构建的标量侧执行/取指或 codegen 差异上。**不能写"已证实是 ICache"**。**2026-11-07 云端 E0 三进程重跑显示该回退只在原始进程出现、两次独立重跑完全消失（ΔB 从 +18.8%/+5.1% 变为 -0.19%/+0.28% 与 +0.07%/+0.19%），判定为单进程伪影；见附录 A 与[调试记录](调试记录-J062-BF16回退-E0E5-20261107.md)。** |
| 社区调研 | 公开资料**没有**"只改入口分派→未改路径变慢"的直接解释；但可提供三条官方机制依据（头开销/核数/kernel 类型自动识别、TPipe 结构改动 -17%、tilingData us 级拷贝）、AIV ICache 16 KB + 官方 `icache_miss_rate` 阈值、以及"布局即性能变量"的通用证据（BOLT 仅重排布局 +8%~15.3%）。910C 无上板 Pipe/InstrTimeline，J062 归因不能靠指令级 timeline。 |
| 优化方向（建议） | **P0** J062 回退判别（先零成本取证 + 同战役补采指标）；**P1** J059 FP32 完整流水提交化（本地 -12.58%）；**P2** 把短 S 改成独立 tiling key，隔离对既有路径的二进制扰动（从机理上消除本次回退风险）；**P3** 头开销专项（显式 kernel 类型/核数、TPipe 重构、tiling 读取），来自本次社区调研的新方向；**P4** FP16 算术优化（暂缓，需数值语义授权）；**P5** J056 继续由用户试交。不建议：无 profile 扫参、合并多候选、重测刷门槛。 |
| 当前阻塞 | **云端已恢复连通**（2026-11-07 10:08，根因是 DSH 文件沙箱阻止 AtomGit 引导脚本写 `C:\Users\lin\.atomgitdevenv`；每次 ssh/scp 需放宽沙箱或用 VS Code 集成终端）。E5/E7/E0 已完成：**J062 的 BF16 S=1 回退被判定为单进程伪影、两次独立重跑未复现**，35 个启用输入在三进程均 210/210 轮全胜。下一步决策：是否按多进程证据更正 J062 记录、是否推进 J062 候选转换（需用户授权）。 |

---

## 1 任务理解与本次动作

用户委托：①阅读项目文件并接手；②去社区调研优化方案；③测试能否成功连接云端；④完成后写调研文档；⑤**先不进行优化**。

据此本次执行：

1. 只读核验两份交接文档、两工程副本、关键实验证据（J035/J056/J059/J060/J061/J062）与共享台账；独立复算 J062 的 19 组失败、函数体计数、官方数字与证据包 SHA256。
2. 并行四路取证（分工产物见第 9 节）：云端连通性实测、昇腾社区/官方资料调研、跨平台通用调研、J062 本地证据链审计。
3. 汇总形成本报告：现状、云端结论、回退审计、调研结论、优化方向、最小判别实验。

**未做**：改代码、建包、跑精度/压力/性能、设备实验、SSH 登录后的任何远端命令、下载远端数据、官方平台操作、git commit/push。

---

## 2 接手现状

### 2.1 两个副本与不可丢弃的工作树

| 位置 | 用途 | HEAD | 工作树 |
|---|---|---|---|
| `C:\Users\lin\Documents\Codex\2026-11-05\task-2\publish-repo` | 最新工程 R | `a68973e72f53499bd5d9eeafc9987c9e4715b5d8` | 195 项未提交/未跟踪（J054 有暂存；J055–J062 未跟踪） |
| `E:\suanzi\cann-ascendc-ops-collab` | 旧副本 E | `fc85a48701b1fb578cd8e5208c9b196a471ba267` | 97 项，实验只到 J053 |

- 个人笔记根 `N = R\people\JadeYara\notes\PReluGradUpdate_2329`；第一入口 `N\CHECKPOINT-20261107.md`。
- 共享 `STATUS.md` 仍写 11 月 4 日 Owner 的 P105 状态，晚于它的 J050–J062 另有记录；**不要用旧 STATUS 否定新证据，也不擅自改共享阶段**。
- 已确认 E 不能覆盖 R；不得 reset/clean、移动旧实验或把工作树当可丢弃产物。

### 2.2 题目语义与不可丢的正确性约束

依据 `TASK.md`、`analysis\SUMMARY.md` 与生产源码：

- `dx = features > 0 ? grads : weights * grads`；`update = features > 0 ? 0 : features * grads`。
- 通道维为 dim1；`weights` 支持 `[1]` 与 `[C]`；**update 不做通道归约**。
- **必须保持 Compare + Select 语义**，不用掩码乘法偷换分支（`0×Inf=NaN` 与 NaN payload 在极值下不等价）。
- BF16 用独立 tiling key，**不能当 half**；低精度链路升 FP32 计算、`CAST_RINT` 回写。
- 不引入 GM 权重展开或无依据 workspace；不把局部 Vector 活动减少等同于完整时延改善。
- 精度口径：有限值按位、±0 等价、NaN 类别相同；历史套件不是 IEEE 特殊值穷尽验证。J057 曾遇到 `0×Inf` 的 NaN payload 与 NumPy 不同，该原始用例不能写成逐位通过。

### 2.3 版本与证据分层（历史数据，本次未复跑）

| 版本 | 范围/主要变化 | 已有证据与结论 |
|---|---|---|
| J035 | FP32 split-C 的 UB 256B 相位布局；冻结基线 | 历史 750 精度、1728 压力；目标 `[127,8192,1]` 12.108→10.795 μs；用户确认官方提交 1148578 五点 Pass：7.15/7.29/7.11/5.88/6.06 μs |
| J056 | resident 多 tile、C=64、S=1 周期权重广播；低精度权重只转换一次 | 独立候选；816/816 回归；目标三 dtype 快约 18.8–19.1%；严格控制 212/216；用户确认官方 1191764 五点 Pass：7.21/7.16/7.19/5.58/5.84 μs |
| J059 | 原 FP32 split-C、tileLen=4096、输入/输出完整双槽流水 | `[127,8192,1]` 11.0275→9.6400 μs（-12.58%）；2820 次正确性；168/168 控制；66/66 启用轮次胜；无官方结果 |
| J060 | J059 基础上合并队列/生命周期 | 相对 J059 9.795→9.230 μs，但控制 151/168；三处未改 BF16 慢点原因未明；仅诊断候选 |
| J061 | 短 S（2–32）跨通道连续 tile、周期 mask 广播 | 2826 次双输出检查；[8,64,3] 三 dtype 提升 38.5%/43.6%/48.2%；控制 255/270；不晋升 |
| J062 | 只修有效 dtype/key 与 `blockFactor>S` 两项入口条件 | 2826 次检查通过；35 个启用输入六轮全部改善；[8,64,3] 提升 38.82%/46.51%/44.76%；**控制 251/270 未通过（BF16 S=1 回退）**；归档断点，不晋升 |

### 2.4 官方数字只能这样引用

- J035 / 1148578：7.15 / 7.29 / 7.11 / 5.88 / 6.06 μs，五点 Pass。
- J056 / 1191764：7.21 / 7.16 / 7.19 / 5.58 / 5.84 μs，五点 Pass。
- 来源是本地记录中的**用户确认**；本次未登录官方平台，也未下载提交快照独立校验源码映射。
- 官方五点 shape/dtype、硬件、CANN 与计时协议未知；不能把它们绑定到本地目标输入，也不能用耗时反推官方 shape。
- 本地收益 ≠ 官方提升；不同进程的绝对耗时不能相减拼成排行榜。

### 2.5 本次独立核验

- 两仓 HEAD、git status 与路径与交接文档一致；J035 生产 workdir 46 文件对冻结 manifest **46/46 匹配**。
- `J062-text-evidence.zip`：331788 字节、71 项、SHA256 `b78d8445…e3007e3`，与 README/交接一致；J061（`01f86b05…`）、J059（`bb1889c9…`）证据包 SHA256 亦与各自 README 一致。
- J062 `control-failures.json` 与 `evidence/perf/summary.json` 中 `controls_stable==false` 记录逐字一致（19 条）；`270=45×6`、`251=270-19`、A/A 最大 0.48 μs。
- 文档主要性能数字逐项对上（见第 4 节）；发现两处口径提醒：README 表头 "A/J035" 实为**本进程内重编的 A 别名**；"J061 FP16 恢复"引用的 6.38→6.37 是 J062 进程内值，J061 当时是 6.47→7.74，两组不能混用。

---

## 3 云端连接实测

完整记录见 `分工产物\cloud-connectivity.md`。结论：**部分连通，不能记为"成功"；可信主机指纹核验未完成。**

### 3.1 分层事实（2026-11-07 09:46–09:49 +08:00）

| 层级 | 结果 | 依据 |
|---|---|---|
| 配置可读、Host 指向本地转发 | 通过（只读） | `C:\Users\lin\.atomgitdevenv\.ssh\config`：Host 别名 `devenvc_…atomgit.0`、HostName `127.0.0.1`、Port `31930`、User `developer`；含 `StrictHostKeyChecking no` 与 `UserKnownHostsFile /dev/null`（风险说明，不是推荐配置） |
| 本地端口监听 | 通过 | `127.0.0.1:31930 LISTENING`，PID 44308（进程名 `Code`，启动 09:22:30） |
| TCP 可达 | 通过 | `Test-NetConnection` 成功；`TcpClient` 15 ms |
| SSH banner / 协议开始 | **失败** | 连接后 5 s、15 s 均读不到任何字节；`ssh-keyscan -T 8` 退出码 1、无输出 |
| alias 方式 SSH | **失败（本地配置门禁）** | `ssh <alias> "exit 0"` 退出码 255；`-vv` 显示 AtomGit `Match originalhost … !exec "cmd.exe /d /c call …bootstrap.cmd"` 命中未通过，改走 `ProxyCommand cmd.exe /d /c exit 1` |
| 主机身份 / 认证 / 远端 exec | **未到达** | 无 banner、无 host key、无认证尝试 |

补充：探测前约 6 分钟（09:40:48–09:40:54）VS Code Remote-SSH 有一次完整成功连接（forwarding server、tunneled port、exec server created、Found running server pid=908），且其 `ssh` 进程（PID 48284）09:47–09:48 仍为 ESTABLISHED。**这不能推出"当前可正常新连接"，也不能推出"云端宕机"。**

### 3.2 可信身份核验状态

- 本次**未取得任何 offered 主机公钥/指纹**，也无可信参照；历史日志里的 `(ED25519)` 只是历史现象（且校验被关闭），不能当证据。
- 11 月 7 日日志中的 `Existing exec server … timed out` 多为缓存重建（随后成功）；`Connection to 127.0.0.1 closed by remote host` 出现在 09:22:59 与 09:32:58，随后自动重连成功。
- **不能**写成"指纹错误导致断连"；本次失败点更早（转发/中继 + bootstrap 门禁）。

### 3.3 恢复最小步骤（只建议，未执行）

1. 用户在可信 AtomGit/云控制台核对实例身份、状态与转发入口，并取得该实例 SSH 主机公钥 SHA256（按算法分别记录）。
2. 在 VS Code 侧触发一次可控重连，观察日志出现 `Starting forwarding server`、`Tunneled port`、`Exec server … created and cached`、`Found running server`；把实际 offered 指纹与第 1 步记录逐项比对。
3. 比对通过前不执行远端命令；通过后先做最小只读检查（`hostname`、`date`、`npu-smi info`、`/home/developer/2329-JadeYara-20261104-1812` 下作业/日志状态），**先查旧作业再决定是否重跑**。
4. 命令行入口应优先用 VS Code 集成终端（继承已建立会话）；不手工建隧道、不改端口/配置、不 kill 进程。
5. 端口会随开发环境重启变化，不要硬编码 31930。

### 3.4 对调研的影响

- 本机是 Windows，未发现可用 CANN 工具链（Docker 服务未运行、WSL 访问受限），**本地无法编译 Ascend C、也无法重建 J061/J062 二进制**。
- 因此所有需要二进制/原始 profile 的取证（见第 7 节 E0/E1/E5）都要在云端或具备 CANN 9.3 的 Linux 环境执行；在云端恢复前，只能做本地文本证据复核。

---

## 4 J062 BF16 回退：证据审计（核心）

完整审计见 `分工产物\j062-evidence-audit.md`。

### 4.1 19 组控制失败应按三类看

判定口径来自 `control\campaign.py`：A1/A2 极差 ≤0.1 μs 且 ≤2% 为 A/A 稳定；未改路径（`modified=false`）还要求 B 落在 A1/A2/C 的极差门槛内。

| 类别 | 组数 | 内容 | 性质 |
|---|---:|---|---|
| ① 系统退化 | **12** | BF16 `[127,64,1]`×6、BF16 `[127,8192,1]`×6 | B 单独偏慢，A1/A2/C 一致；六轮恒定 |
| ② 临界越界 | 2 | BF16 `[4,7,33]`（S=33）第 3/4 轮 +0.11~0.12 μs（+2.07%~2.26%） | 卡在 0.1 μs + 2% 门口；整案 +1.88% |
| ③ 稳定性越界 | 5 | FP16 `[257,257,3]`×4、FP16 `[2048,9,5]`×1 | A/A 或 A/C 漂移 0.12~0.48 μs；B 本身 6/6 大胜，**不是回退** |

具体退化幅度（六轮中位数）：

| dtype/shape | A1/A2（J035 别名） | C（J035 第二别名） | B（J062） | ΔB | 六轮 |
|---|---:|---:|---:|---:|---|
| BF16 `[127,64,1]` | 6.38 / 6.38 | 6.40 | **7.58** | +1.20（+18.81%） | 全慢 |
| BF16 `[127,8192,1]` | 12.00 / 12.00 | 12.04 | **12.62** | +0.62（+5.10%） | 全慢 |
| BF16 `[4,7,33]` | 5.32 / 5.32 | 5.32 | 5.42~5.44 | +0.10~0.12 | 2/6 越界 |

### 4.2 新发现：回退的引擎指标签名

`evidence/perf/summary.json` 的 1080 条 window 记录含 6 个指标。本次独立复算（A1/A2/C 六轮均值 vs B 六轮均值，单位 μs）：

| case | 指标 | A1/A2/C | B | Δ |
|---|---|---:|---:|---:|
| BF16 `[127,64,1]` | Task Duration | 6.38 | 7.58 | **+1.20** |
| | aiv_time | 4.16 | 5.51 | +1.35 |
| | **aiv_scalar_time** | **2.29** | **3.45** | **+1.16** |
| | aiv_vec_time | 0.24 | 0.24 | **0.00** |
| | aiv_mte2_time | 2.00 | 2.25 | +0.25 |
| | aiv_mte3_time | 0.33 | 0.27 | -0.06 |
| BF16 `[127,8192,1]` | Task Duration | 12.01 | 12.62 | **+0.61** |
| | **aiv_scalar_time** | **4.85** | **6.05** | **+1.20** |
| | aiv_vec_time | 4.10 | 4.10 | **0.00** |
| | aiv_mte2_time | 3.58 | 3.18 | -0.40 |
| BF16 `[4,7,33]` | Task Duration | 5.32 | 5.42 | +0.10 |
| | **aiv_scalar_time** | **2.49** | **3.02** | **+0.53** |
| | aiv_vec_time | 0.39 | 0.39 | 0.00 |
| J061 FP16 `[127,64,1]`（历史对照） | aiv_scalar_time | 2.11 | 3.26 | **+1.15** |
| J061 FP16 `[127,8192,1]`（历史对照） | aiv_scalar_time | 4.70 | 5.84 | **+1.15** |

要点：

- **回退几乎全部落在 AIV scalar 时间；vec 时间逐轮精确不变**；其余 7 个未改路径案例 `|Δscalar| ≤ 0.026 μs`。
- 两个 S=1 形状元素数相差约 128 倍（8128 vs 1040384），但 scalar 增量都约 **+1.15~1.20 μs**，更像**每-launch 的固定标量开销**，而不是随数据量增长的工作量；
- J061 的 FP16 S=1 失败有**完全相同的签名**——"慢点"在两个构建间从 FP16 换到了 BF16。这是"每次构建换一个不幸运的 dtype"的直接迹象。
- 本地也能排除两个混杂：`block_nums` 在 A/B/C 完全一致（case 17=40、case 21=40、case 40=29），五张量地址在 A/B/C 完全一致。**不是分核/地址分配差异。**

### 4.3 源码改动与"未改路径"的边界

`dispatch-only.diff` 相对 J061 只改 `op_kernel/p_relu_grad_update_b.cpp` 两处：

1. `if constexpr (schMode < 4)` → 只匹配 3 个有效默认 key 且要求 `DTYPE_GRADS` 与 schMode 同 dtype；
2. 短 S 运行期条件追加 `blockFactor > innerSize`。

两项都是**使进入 ShortS 的集合变小**：S=1 输入在两版都因 `innerSize > 1` 不成立而不进入 ShortS；Host 的 `vectorWeights`（S=1）分支在 J035/J062 间无差异（Host 唯一差异在 `shortWeights` 分支）；`blockFactor` 条件实际只把 FP32 `[4,6,8]` 由启用变停用（36→35）。

**因此源码级可排除"BF16 S=1 因条件变化走入另一实现"**；但编译产物确实变了（见 4.4），变化发生在 TU 级 codegen/布局，而不是运行时分派语义。运行级仍缺 `tiling key`/kernel symbol 回读日志（`overview.tiling` 是本地推导，不是设备回读）。

### 4.4 二进制与布局事实（已证实"变了"，未证实"导致性能"）

- 每个 dtype 的 `.o` 含 15 个 tiling-key 入口函数；135 = 3 别名 × 3 dtype × 15 入口。
- 与 J053 基线比较：A 45/45、C 45/45 一致（**间接证据**：本地只有脚本布尔，缺基线与 `.o`），B 30/45。
- B 的 15 个不同：3 个 default 入口因实例化 ShortS 变长（f32 3976→8512、f16 4648→10116、bf16 4468→9568），12 个 fallback 入口（S1/MERGED/RESIDENT/SPLITC 各 dtype）**同长度但字节不同**。
- J061→J062 对比：6 个无效 key 的 stub 由 4740/5304/5672 B 缩回 128 B；3 个 default 各 +8 B；**12 个 fallback 入口再次"同长不同 hash"**。fallback 入口地址在 `.o` 内回移约 **9~11 KiB**；J062 的 B 相对 A 整体偏移 f32 +4536 B、f16 +5468 B、bf16 +5100 B。
- 注意：`bodies.sh` 比对的是符号字节区间 sha256，**不比较重定位、不做反汇编、无 unmatched 逻辑**；"同长不同字节"既可能是重定位位移，也可能是真实指令差异——**未验证**。

### 4.5 历史先例（仅相关，不能当因果）

- **J060**：FP16/BF16 整个 `.o` 与 J059 **逐字节完全一致（含 section 内容/偏移/地址）**，仍有 3 个未改 BF16 输入（`[8,64,3]`、`[2,32,17]`、`[127,8192,1]`）+0.14~0.24 μs 慢点。说明"未改路径变慢"可以在相关 `.o` 完全一致时发生，指向别名/加载/测量层，而非算法。
- **J056/J057/J059**：也存在"未改动 FP32 回退函数长度相同但字节不同""低精度路径顺序波动"的记录；这是该工程长期存在的现象族。
- **J021/J022（CANN 9.0.0 时代）**：本算子存在真实的指令缓存/代码布局敏感现象——目标 tiny 预热可消除回退、ICache miss 与 AIV cycles 同步升高、`ICachePreLoad(4)` 使冷态 tiny 改善 0.7~0.9 μs；但那次是**冷启动/顺序效应**（可被预热消除），与 J062 **逐轮恒定**的 ~1.2 μs 标量开销**不是同一种证据**，只能作"可检验的类比"。
- 生产基线 J035 本身已包含 `ICachePreLoad(4)`（S1 入口 5/6），说明预取是既有设计，不是新增手段。

### 4.6 假设台账

| 假设 | 支持 | 反对/矛盾 | 当前判定 |
|---|---|---|---|
| **H1 该构建低精度 fallback 实例的指令放置/取指开销**（含 default 入口 +5 KB 造成地址迁移） | scalar 唯一上升、vec 不变、逐轮恒定；慢点在 J061/J062 间换 dtype；J060 `.o` 全同先例；J021/J022 证明本算子对取指敏感 | 缺 ICache/cycles/加载地址/反汇编；两个 S=1 形状 scalar 增量近乎相同（固定成本）也与"局部布局"不完全吻合；J060 的 `.o` 全同先例其实更像加载/别名层 | **最可疑，未验证** |
| **H2 同源码 codegen 差异导致 bf16 fallback 标量指令变多** | 12 个同长不同字节 body 提示 codegen 输出确实变了 | 同长度；f16 有同型变化却不慢；无指令级证据 | 可能，未验证 |
| **H3 别名/加载/测量层环境因素** | J060 强先例（`.o` 全同仍慢）；不同别名是不同二进制/不同加载位置 | 本轮引擎级指标稳定到 0.005 μs，纯噪声解释不了 ±1.2 μs；A/C 两个相同副本之间完全一致 | 部分可能，未验证 |
| **H4 BF16 S=1 实际走了另一模板实例** | 编译期 key 集合变了 | 源码级 host 分支未改、短 S 条件不可达；缺运行期 key 日志 | **基本排除（源码级），待运行日志确认** |
| **H5 UB 预算/tiling 参数变化** | — | J061/J062 对 `vectorWeights` 的 `bytesPerElement`、`tileLen`、key 规则与 J035 相同；block_nums 一致 | 基本排除 |

**结论**：回退是真实的、与该构建二进制绑定的标量侧现象；**根因未定位，不能写"已证实是 ICache/编译布局"**。

---

## 5 社区与跨平台调研结论

完整材料：`分工产物\ascend-community.md`（30 条来源）、`分工产物\general-research.md`（21 条来源）。以下是可支撑决策的要点，均标注证据强度。

### 5.1 昇腾官方资料（910C 相关）

| # | 结论 | 强度 |
|---|---|---|
| A1 | 官方文档承认"算法没改、耗时却变"的三类机制：①**头开销**含核启动、核取址 TLB MISS、同地址冲突、变量资源初始化，随核数增加；**Kernel 类型自动识别依赖编译优化结果、可能失效**，官方建议手动设置；②`TPipe` 在算子类内构造会阻断 Scalar 常量折叠/传播，官方实测 281→236 μs（**-17%**），scalar 占比 21%→17%；③`GET_TILING_DATA` 把 tiling 从 GM 拷到栈，**拷贝 us 级、小 shape 更明显**。 | 强（官方文档） |
| A2 | 与本地**同 SoC `Ascend910_9362`（40 AIV）** 的官方知识库记录：block 数从 32 改到覆盖 40 核后整体时间比 0.758，且"接近启动开销主导的 case 可能没有收益，一个小 workload 已实测回退"——分核/头开销类改动会让部分小 shape 反向变慢。 | 中强（官方知识库 stable） |
| A3 | **AIV ICache 16 KB / AIC 32 KB**（AscendC `ICachePreLoad` 文档，2 KB 粒度）；`asc_get_icache_preload_status` 明确 **Atlas A3 系列支持**；官方 ops-profiling 规范把 `icache_miss_rate` 作为一等指标（<5% 达标、5–15% 警告、>15% 判"代码量过大"）。 | 强（容量/A3 支持）；中（910C ICache 组织未公开） |
| A4 | **910C 没有上板 PipeTimeline/InstrTimeline**（官方 msopprof 文档：仅 950PR&950DT）；A3 可用 Default 指标、Occupancy、Roofline、Cache 热力图、`TimelineDetail`（仿真，限 PyTorch 单算子 API）。`Task Duration`、`OpBasicInfo`（含 block dim）可得。**J062 归因不能依赖指令级 timeline。** | 强（官方文档） |
| A5 | 官方 `ops-nn/activation/p_relu_grad_update` 存在，语义与本题逐字一致（双输出同形、`update` 无归约、dtype FLOAT/FLOAT16/BFLOAT16），入口也是 `template <schMode>` + BroadcastSch 分派；但 README 产品表写"仅 950PR&950DT 支持，Atlas A3 ×"，同仓 `aclnnPreluBackward.md` 却写"A3 支持"，**官方口径矛盾**，需官方确认 910C 路径。另有 `p_relu_grad_reduce`（weight 归约变体）与 `PReluGradFusionPass`。 | 强（代码/文档事实）；口径矛盾需确认 |
| A6 | 官方知识库同构案例：把 gate 内 setup 移入 gate 分支后 device task time 变化，作者明确写"变是确定的，**为什么（簿记移除 vs codegen/layout 副作用）未定**"，并要求先隔离 setup 与结构改动。另有：int64 修复导致 ~6% 稳定回退；近空 kernel 成本约工作 kernel 的 98%（950）。 | 中（draft/单次 A/B，950） |
| A7 | 公开渠道**未找到** CANN 9.3.0-beta.1 / msopprof 26.3 的 release notes 与已知问题列表；未找到 910C 上 kernel 体积/布局 ↔ 性能的量化数据；未找到 hiascend 论坛/知乎/CSDN 上"改一处、别的路径变慢"的可引用实测帖。 | 明确的"无证据" |
| A8 | 测量口径风险有官方记录：上板首跑比稳态中位数慢 **+24.0%~+69.4%**，TTK `bin_perf_us` 丢弃首跑取中位数；官方建议 `msprof op --warm-up=10 --launch-count=5` 并核对 `Current Freq`/`Rated Freq`。 | 强（方法学） |

### 5.2 跨平台通用资料

| # | 结论 | 强度/可迁移性 |
|---|---|---|
| G1 | PyTorch CUDA `prelu_backward_kernel` 就是逐元素 Compare+Select（`mask ? grad : weight*grad` / `mask ? 0 : input*grad`），但 weight grad 额外做 N+空间维归约；本题 `update` 是"归约前 collector"，**不能用框架语义背书，只能由 CANN op_api 契约自证**。PyTorch PR #91238 记录了"掩码乘法→Select"的刻意迁移；issue #156212 实证 `torch.where` 反向 `0×Inf=NaN`。 | 强（源码/issue） |
| G2 | "布局即性能变量"的公开证据：BOLT 仅做 post-link 代码布局重排，在已有 FDO+LTO 的数据中心应用上仍有最高 **8.0%** 提升，对编译器二进制最高 **15.3%**；NVIDIA 论坛实测 GPU I-Cache 容量 miss 约 **3%**，最坏情形是子程序按地址冲突。 | 强/中间接（CPU/GPU，不能外推 910C） |
| G3 | 公开资料**没有**"同文件改入口分派→未改路径变慢"的实测案例；本题现象必须自证。18.8% 高于公开纯 I-Cache 效应，若由取指解释需叠加小 shape 头开销等机制；5.1% 与公开量级相容。 | 明确的"无证据" |
| G4 | 方法学：预热、多次 launch 取中位数、频率核对、随机交错（可降 run-to-run 方差约 40%）、跨进程复现；团队现有"六序平衡 + 控制组 ≤0.1 μs 且 ≤2%"方向正确，缺**跨进程复现**与**频率/DVFS 证据**。 | 强（工具文档） |
| G5 | Select 的 selMode 1/2 在 A2/A3 上需预留 8 KB UB 作为临时区；本项目使用 `SELMODE::VSEL_TENSOR_TENSOR_MODE` + Compare 掩码张量。入口改动是否会改变 selMode 属未验证推测。 | 中（文档；未核本项目编译产物） |

### 5.3 可迁移优化手段清单（对应本项目）

| # | 手段 | 本项目状态 | 建议 |
|---|---|---|---|
| M1 | 显式设置 Kernel 类型与核数（避免自动识别失效；µs 级算子减少启动核数） | **未做**；当前依赖自动识别，Host 已固定 blockDim | 值得单独立项验证（新方向，官方依据强） |
| M2 | 精简 TilingData / 直接按入参指针读取 tiling，避免整块 `GET_TILING_DATA` 拷贝 | **未做**；当前用 `GET_TILING_DATA_WITH_STRUCT` | 小 shape 有官方收益依据；改动触及入口，需与 J062 回退诊断分开 |
| M3 | 避免 `TPipe` 在算子类内构造/初始化；静态创建 LocalTensor、局部变量替代成员 | **未做**；当前 5 个 kernel 类都有 `TPipe pipe_` 成员 | 官方实测 -17% scalar；与算子结构强相关，建议作为独立候选 |
| M4 | Double Buffer / 多 tile 合批 | 已做（J059 双槽流水 -12.58%；J056 广播；J061 短 S） | 沿用，不再无 profile 扩参 |
| M5 | 核间负载均衡（尾核/尾块） | 部分（split-C/多 tile 分核） | 可在 P3 头开销专项中一起看 |
| M6 | DCI 编译选项减尾开销（`--cce-no-dcache-flush`） | 未做 | **官方限 950PR&950DT**，910C 不可默认迁移 |
| M7 | ICache 预取 / 代码体量控制 | 已用 `ICachePreLoad(4)`（S1 入口）；dtype 裁剪已做（J023/J062） | 预取是双刃剑（社区：代码体量增加可能反而触发 miss）；可作判别实验正向对照 |
| M8 | msopprof 诊断（空 kernel 标定头开销、`icache_miss_rate`、Cache 热力图） | 用过部分指标 | 910C 无指令 timeline；先取原始 profile 列 |
| M9 | Compare+Select / BF16 独立 key / 升 FP32 + CAST_RINT | 已坚持 | 维持，作为评审依据 |
| M10 | 改一个实例后审计所有相似实例（多 dtype/多分支） | 有 A/B/C 一致性检查 | 建议固化为脚本：函数体 hash + 地址 + key 映射 + 参数清单 |

---

## 6 优化方向与优先级（建议，未执行）

排序原则：先解掉"使 J062 无法晋升的唯一阻塞"，再选已有最完整证据的方向；每一项都写明所需授权与停损。

| 优先级 | 方向 | 依据与预期 | 风险 | 需要的授权/条件 |
|---|---|---|---|---|
| **P0（已完成 E0/E5）** | **J062 BF16 回退判别** | 2026-11-07 已完成：E5 原始 profile 指标 → ICache miss 无升高、cycles 与 scalar 同步升；E7 机器码 → 执行函数仅差 1 个立即数 word；E0 三进程 → 回退只在原始进程出现、两次重跑消失。判定为单进程运行时/放置伪影 | 仍需多进程统计口径确认；J062 记录需更正（待授权） | 若推进 J062：更正台账 + 正式候选转换 + 完整回归；官方提交由用户操作 |
| **P1** | **J059 FP32 完整流水提交化** | 本地 `[127,8192,1]` -12.58%，2820 精度、168/168 控制、66/66 启用轮胜，是当前证据最完整的独立候选 | 测试工程不能直接当提交包；需要候选转换、别名还原、Host/数学链保持、完整回归与归档；官方 shape 未知 | 代码工作 + 一次完整回归/设备验证；官方提交仍由用户本人操作 |
| **P2** | **短 S 独立 key 化**（把 ShortS 放进新 schMode，而不是塞进 default 入口的运行时分支） | J061/J062 已证明短 S 收益巨大（[8,64,3] 三 dtype 提升 38–48%）；独立 key 可让既有 fallback 入口保持小体积、降低 J062 这类"入口膨胀→地址迁移"风险 | 需改 Host key 选择与 kernel 模板；需要完整回归；收益依赖官方 shape | 立项授权；建议在 P0 结论后做，避免与判别实验叠加 |
| **P3** | **头开销专项**（显式 kernel 类型/核数、TPipe 移出类成员、精简 tiling 读取） | 本次社区调研新增方向，官方有量化依据（-17% scalar、us 级 tiling 拷贝、同 SoC 分核记录）；与 scalar 时间签名直接相关 | 触及 kernel 结构/入口，改动面大；可能与 J062 回退因果混淆 | 需单独立项与 profile 依据；先做离线/仿真可行性评估 |
| **P4** | **FP16 算术优化**（未开始的新路线） | 可能减少 Cast/中间 FP32 缓冲；但数值语义需单独评估 | 与"有限值按位、±0 等价、NaN 类别相同"的精度口径冲突风险高；可能无法过 Judge 阈值 | 先做数值语义论证与最小用例，用户单独授权 |
| **P5** | **J056 继续由用户试交** | 已有一个官方五点和本地 18.8–19.1% 收益记录 | 官方 shape 未知；严格控制在 212/216 略超门槛 | 用户本人操作 |
| — | **不建议** | 无 profile 扫 shape/tile/核数；把 J056+J059+J062 合并；无新证据重开 J046/J054/J057；重测刷控制门槛；用掩码乘法/update 归约/bf16 当 half | — | — |

**关于"官方收益"的现实判断**：J035 与 J056 的官方五点差异很小（均值约 6.70 → 6.60 μs，且点 1/3 略退），说明官方平台上几个百分点的本地收益未必线性兑现；而 J062 这类 38–95% 的本地大收益，其官方可迁移性完全未知（官方 shape 未知），且当前被 BF16 S=1 回退阻塞。**先解阻塞，再谈提交。**

---

## 7 最小判别实验方案（只设计，未执行）

共同固定项（所有实验一致）：

- **对照**：A1/A2 = 冻结 J035 别名（两次同接口测量），C = 冻结 J035 第二别名；B = 待测变体；同一进程、同一 stream、每 case 五个固定张量地址。
- **输入**：只跑 45 输入中的 10 个 `modified=false` fallback 输入（三 dtype × `[4,7,33]`、`[127,64,1]`、`[127,8192,1]`）+ FP32 `[4,6,8]`；不扫 shape/核数。
- **协议**：六序 `ABCA/CAAB/BACA/ACBA/BAAC/CABA`；每窗 350 次、舍 200、取 150 完整 Task 中位数；`≤0.1 μs 且 ≤2%` 双门槛不变；precision 2826 调用必须先 PASS；不重测刷门槛。
- **预注册判据**：每个实验的"支持/排除"在开跑前写死；最多 3 次进程重启；达到判据立即停。

| 编号 | 唯一变量 | 对照 | 预期观察（支持/排除） | 停止条件 | 前置条件 |
|---|---|---|---|---|---|
| **E0** | 无代码、无重编：同一 J062 包再跑一次；每 case 把 B 放两个独立窗口（B1/B2）并重启一次；同时取回远端原始 profile 全列 | 现有 A/C/B | B1≈B2 且重启后复现 → 排除"一次性环境/顺序"；B1≠B2 → 测量/顺序；原始 CSV 中 `icache_miss`/`cycles`/`instr` 列有差异 → 支持取指/指令数 | 1 次重启 + 2 个 B 窗口即停 | 云端恢复；**最高性价比**，可能无需新实验 |
| **E5** | 只补采指标（不改码）：`icache_miss_rate`、aiv cycles、scalar/vec 指令计数、实际 kernel symbol/tiling key | J062 A/C/B | miss 显著升高 → 取指假设；cycles/指令数升高而 miss 不变 → codegen/指令数；两者不变 → 别名/加载/测量 | 复用 E0 战役 | 云端 + 可用指标列 |
| **E1** | 只隔离代码布局：J062 源不动，用 `noinline` 包装或显式对齐填充改变 fallback 入口地址，入口条件与 kernel 类不变 | 旧布局 J062-B | bf16 S=1 的 Δscalar 从 ~1.2 μs 明显缩小/消失 → 支持布局/取指；仍 ~1.2 μs → 转 codegen/测量；慢点搬到别的 dtype → 支持"放置敏感" | 一次构建；Δscalar <0.1 或保持 >1.0 即判 | 云端构建 |
| **E2** | 只改 BF16 分派 key：A 与 B **同时**改到同一合法 key（如把 `[127,64,1]` 从 MERGED 改到 S1/RESIDENT） | 同 key 的 A/B；另留旧 key B 参照 | 慢点随 key 迁移消失 → 绑定特定实例/入口；两个 key 下 B 都慢 → 与实例无关；A 在新 key 也慢 → key 本身是因素 | 一次构建，仅测 bf16/fp16 两形状 | 云端构建 |
| **E3** | 只改 `blockFactor` 条件一项（J062 去掉该条件，其余保持） | J062-B | 预期 BF16 S=1 完全不变（S=1 不经过该条件）→ 排除第二项修正是慢点来源 | Δ<0.05 μs 即判 | 云端构建 |
| **E4** | 只改 dtype 匹配条件一项（恢复 J061 的 `schMode < 4`，保留 blockFactor 条件） | J062-B | 6 个 stub 重新膨胀后 BF16 慢点幅度/位置变化 → 支持"编译膨胀/布局"链条；不变 → 与膨胀无关；同时看 fp16 是否复现 J061 慢点 | 一次构建 | 云端构建 |
| **E6** | 只改 B 在进程内的位置（B 连续两次、B 放首/末） | A1/A2/C 不变 | B 慢点随位置变化 → 顺序/上下文效应；与位置无关 → 该二进制的稳定属性 | 测量层，一次即可 | 云端 |
| **E7**（可选） | 只加极小 S=1 输入（如 `[1,64,1]`、`[7,64,1]` bf16） | 现有两 S=1 形状 | 若 scalar 增量仍≈1.2 μs → 固定每-launch 成本（支持头开销/取指/入口固定开销）；若随规模缩小 → 数据相关工作 | 2~3 个输入即停 | 云端 |

**首选顺序**：E0（零成本取证 + 复现性）→ E5（同战役补 ICache/cycles，直接分离"取指"与"指令数"）→ E1（只动布局）。E2/E3/E4 在 E0/E5 给出方向后再选其一，避免同时动两个因素。

**最高性价比的缺失数据**（云端恢复后第一件事）：

1. `/home/developer/2329-JadeYara-20261104-1812/J062/perf/profile` 的原始 `op_summary*.csv`（本地只有 6 指标中位数；ICache/cycles/指令数可能已在原始 CSV 中）；
2. J062 的 `.o` 本体与最终包（用于反汇编判定 12 个"同长不同字节"是 reloc 还是真实指令差异；计算最终地址/对齐）；
3. kernel 选择记录（tiling key / kernel symbol / blockDim）——目前只是源码推导；
4. 基线 `J053-function-bodies.json`（远端约 59 KB）——本地无法重算 A/C 的 before hash；
5. J061/J060 的原始 profile 与 `.o`（跨构建对照：慢点从 FP16 换到 BF16 的机制）。

---

## 8 未运行事项与风险声明

**未运行**：CANN 构建、精度/压力测试、性能/设备实验、模拟器、官方平台登录/提交、远端 SSH 命令、远端数据下载、git commit/push。本次只做本地只读读取、JSON/日志解析、sha256 复算与网页检索。

**风险与边界**：

1. 本报告的"云端"结论只来自本机只读探测；没有恢复连接，也没有验证远端实例身份与作业状态。
2. 历史性能/精度数字均来自既有本地记录，本次未复跑；J062 回退的判别结论尚未产生。
3. 官方成绩只按用户确认记录引用，未独立核验提交快照、官方 shape 或计时协议。
4. 本地微基准受"布局彩票"影响：J060 的 `.o` 全同先例与 J062 的 dtype 换位说明**同一源码不同构建可能改变某形状的 µs 级表现**；官方平台会重新编译，本地大收益不能直接当官方收益预期。
5. 不同进程、不同 CANN 版本（9.0 vs 9.3）、不同副本（R/E）的绝对耗时不能混算。
6. 本机无 CANN 工具链，二进制级取证必须依赖云端或另建 Linux 环境；本报告不替代云端验收。

**下一步决策点（需要用户确认）**：

1. 是否按第 3.3 节恢复云端（含可信指纹核验授权）？
2. 云端恢复后，是否先做 E0/E5（只读 + 补采，不建新包）？
3. P0 判别实验完成后，选择"修 J062 布局/结构"还是"隔离短 S 为独立 key"？
4. P1（J059 提交化）是否与 P0 并行立项？
5. P3（头开销专项）与 P4（FP16 算术）是否进入候选池？

---

## 9 引用材料与分工产物

### 9.1 交接与项目材料

| 材料 | 路径 |
|---|---|
| 项目方向与调研交接 | `C:\Users\lin\.dsh\attachments\v1\files\5f\5fbefe43…\01-项目方向与调研交接-PReluGradUpdate_2329-20261107.md`（SHA256 `5fbefe43…`） |
| 云端连接与调试交接 | `C:\Users\lin\.dsh\attachments\v1\files\5a\5acb33c1…\02-云端连接与调试交接.md`（SHA256 `5acb33c1…`） |
| 最新工程 R | `C:\Users\lin\Documents\Codex\2026-11-05\task-2\publish-repo`（HEAD `a68973e…`） |
| 个人笔记 N | `R\people\JadeYara\notes\PReluGradUpdate_2329`（入口 `CHECKPOINT-20261107.md`） |
| 旧副本 E | `E:\suanzi\cann-ascendc-ops-collab`（HEAD `fc85a48…`） |

### 9.2 本次四份分工产物

| 文件 | 内容 |
|---|---|
| `分工产物\cloud-connectivity.md` | 云端连通性只读实测：分层事实、命令/退出码、日志片段、未做与恢复步骤 |
| `分工产物\ascend-community.md` | 昇腾社区/官方资料调研：30 条来源、H1–H7 证据对照、M1–M10 手段、E0–E4 与 O1–O10 观察清单 |
| `分工产物\general-research.md` | 跨平台调研：21 条来源、PReLU 反向实现对照、I-Cache/布局证据、微基准方法学 |
| `分工产物\j062-evidence-audit.md` | J062 证据链审计：19 组三类拆分、函数体口径、diff 语义、引擎指标新发现、E0–E6 实验、缺失数据清单 |

### 9.3 社区调研精选来源（完整清单见分工产物）

- Ascend C 头尾开销优化：`https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/operator_practice/simd_operator_optimization/overhead_optimization/`
- AIV/AIC ICache 容量与预取：`https://www.hiascend.com:6066/doc_center/source/zh/CANNCommunityEdition/80RC3alpha003/apiref/opdevgapi/atlasascendc_api_07_0266.html`
- A3 ICache 预取状态支持：`https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/c_api/cache_ctrl/asc_get_icache_preload_status.md`
- msopprof 用户指南（910C 无 Pipe/InstrTimeline）：`https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md`
- 官方 PReLU 反向算子（ops-nn）：`https://atomgit.com/cann/ops-nn/blob/master/activation/p_relu_grad_update/README.md`
- PyTorch PReLU backward 内核：`https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationPreluKernel.cu`
- PyTorch `0×Inf` issue：`https://github.com/pytorch/pytorch/issues/156212`
- BOLT（布局重排）：`https://arxiv.org/abs/1807.06735v1`
- Google Benchmark 方法学：`https://google.github.io/benchmark/user_guide.html`

### 9.4 本次独立复算结果

- `J062-text-evidence.zip` SHA256 `b78d8445545c69d1bca2a8d8687da0092feacbd7235b3221f5ee45a71e3007e3`，71 项，331788 字节。
- `J061-text-evidence.zip` SHA256 `01f86b05041d80e1abf6db72c5f665c31d7f4352f83a18ab2da76c2d13152a64`；`J059-text-evidence.zip` SHA256 `bb1889c9a975451054b17a256f07a0501fe4ffd23a803e01950eb09649682e21`。
- J035 生产 workdir 46 文件对 `J035\source-sha256.json` 46/46 匹配。
- J062 19 组控制失败与 `summary.json` 逐字一致；A1/A2/C 与 B 的 6 指标复算见 4.2。
- `block_nums`（case 17=40、21=40、40=29、20=39）与五张量地址在 A/B/C 完全一致。

---

*本报告为调研交付，不构成优化授权、官方提交或云端操作授权；后续任何代码、设备、平台或连接操作，以用户逐项指令为准。*

---

## 附录 A：2026-11-07 云端调试补充（E5 / E7 / E0）

详细记录见[调试记录-J062-BF16回退-E0E5-20261107.md](调试记录-J062-BF16回退-E0E5-20261107.md)。

### A.1 云端连接已恢复

- **根因**：DSH `workspace-write` 文件沙箱拒绝 AtomGit `forward-bootstrap` helper 写 `C:\Users\lin\.atomgitdevenv` 的锁文件/日志 → ssh 门禁 `Match exec` 每次失败。放宽一次文件沙箱后 bootstrap 立即 `forward.ready`，ssh 连接成功。
- 现场：hostname=`306e9d…`（devEnvId）、`developer`、aarch64、40 核、CANN `cann-9.3.0`、`npu-smi 25.5.5` 两 chip Health OK 且无运行进程。
- 主机公钥：offered `SHA256:qPhIpfPEmRfFhfArnWWBP/aRrj0JXj7vr70EAlwK4i0` 与远端 `/etc/ssh/ssh_host_ed25519_key.pub` 指纹一致；**可信参照仍需用户在控制台核对**。
- 注意：当前沙箱下每次新 `ssh/scp` 都需同样放宽，或改用 VS Code 集成终端。

### A.2 E5：ICache/cycles（原始 J062 profile）

- `op_summary` 46 列、378000 行；`aiv_icache_miss_rate` 稳态 ~0.00003–0.00014（全表非零仅 689 行、max 0.164，集中在预热首样本）。
- 慢的 B 行：`aiv_scalar_time` +1.15~1.20 µs、`aiv_vec_time` 精确不变、`aiv_total_cycles` +32%/+8%；**ICache miss 与 A/C 同 0 量级 → 无 ICache miss 升高证据**。
- `block_nums` 与五张量地址在 A/B/C 完全一致。

### A.3 E7：机器码对比

- S=1 输入执行的 MERGED 函数：A vs B 仅差 **2 字节 / 1 个 4B word**（立即数/相对地址，随函数在 `.text` 内平移）；S1/RESIDENT/SPLITC 同样。bf16 与 fp16 同型。
- **排除**"J062 让未改路径指令数变多/codegen 恶化"的解释。

### A.4 E0：三个独立进程（同包、同 plan、同 msprof 参数）

| 进程 | 控制失败 | BF16 `[127,64,1]` ΔB | BF16 `[127,8192,1]` ΔB | BF16 `[8,64,3]` 收益 | 35 启用输入 |
|---|---:|---:|---:|---:|---:|
| P1（原始战役） | 19 | **+18.79%** | **+5.10%** | -44.76% | 210/210 轮胜 |
| P2（10:19 重跑） | 6 | **-0.19%** | **+0.28%** | -48.64% | 210/210 轮胜 |
| P3（10:24 重跑） | 4 | **+0.07%** | **+0.19%** | -48.00% | 210/210 轮胜 |

- P2/P3 的控制失败**全部是 FP16 长输入的 A/A 绝对漂移**（`[257,257,3]` 677 µs、`[2048,9,5]` 193 µs 等，0.12–0.48 µs，相对 <0.1%），不是候选回退。
- P2/P3 中 BF16 目标形状的 `aiv_total_cycles` 与 A/C 完全一致（P1 的 +32% 消失）。
- 跨进程漂移量级：A 基线极差 mean 0.213 / max 0.616 µs；B 极差 mean 0.343 / max 1.423 µs（即 P1 的异常 case），与历史"跨进程 A/A 最大 1.041 µs"一致。

### A.5 更新后的判读

1. **J062 的 BF16 S=1 "系统性回退" 是单进程运行时/放置伪影**，两次独立重跑完全消失；不能作为否定 J062 算法收益的证据。
2. J062 的**启用输入收益稳健**：三个进程均 35/35 输入、210/210 轮全胜（`[8,64,3]` 约 -45%~-49%）。
3. H1（指令放置/取指）需改写为"per-process 加载/放置/首次执行状态"，**不是稳态 ICache miss**；H2（codegen）与 H4（key 路由）、H5（tiling）均已排除；H3（别名/加载/测量层）得到支持。
4. 现行"≤0.1 µs 且 ≤2%"控制门槛对长 kernel 的 A/A 漂移过于严格（相对 <0.1% 仍越界），属**测量口径限制**；本轮未改门槛、未重测刷过，交由 Owner/用户决策。
5. J062 README/DEC、J060/J061 的"未改路径慢点"记录建议按"多进程复现性待核/伪影"更正；**待用户授权后才写入 R 仓库台账**。
