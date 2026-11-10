# 道路台账 — PReluGradUpdate_2329

> 开新路前先搜本表与 `tasks/archive/**/path-ledger.md`。  
> 规范：`docs/collaboration/01-path-ledger.md` · 性能止损：`docs/collaboration/02-perf-gate.md`

| ID | 日期 | 作者 | 阶段 | 假设 | 方法 | 时限 | 结果 | 结论 | 耗时 |
|----|------|------|------|------|------|------|------|------|------|
| P001 | 2026-03-22 | team | 协作 | 先立规范与分类可降低后期重复找路成本 | 建 GitHub 协作仓：原则/台账/止损/题目卡；**不开发算子** | 本迭代 | 仓与规范落地 | 采纳 | — |
| P002 | 2026-03-22 | team | 准备 | 官方模板含 dtype/工程骨架，先同步再设计 | 自 `D:\华为算子比赛\` 同步 ZIP → `materials/` 并解压阅读 | 0.5d | Kernel/Tiling TODO；语义见 reference.py | 采纳 | — |
| P003 | 2026-03-22 | team | 分析 | 开题前 ≥3 AI 分析可降低错误优化方向 | AI-1/2/3 独立成文 → `analysis/SUMMARY.md` 定稿；主路径=行片切分+单趟融合+bf16独立key | 本迭代 | 三路齐；裁决见 SUMMARY | 采纳 | — |
| P004 | — | — | 设计 | 按 SUMMARY P0 做最简正确实现可达功能+预期全核带宽 | design.md：InferShape；TilingData(C/inner/ROW_SLICE)；四队列；逻辑GT；MODE_2 bf16 | 设计评审前 | （待做） | 规划中 | — |
| P005 | 2026-03-22 | team | 协作 | AI 无围栏会串题/跳阶段/搞乱目录 | 落地 `AGENTS.md` + `05-ai-work-protocol` + Cursor rule + 会话卡片 | 本迭代 | 规范已合入协作仓 | 采纳 | — |
| P006 | 2026-11-02 | team | 分析 | 本地 Codex/Claude 设计比 Cursor 粗案更细 | 导入 `PReluGradUpdate_problem_2329_plan`；SUMMARY 以 design_final 为实施主依据（Select/Inf/模式R） | 本迭代 | 已合并定稿 | 采纳 | — |
| P007 | 2026-11-02 | team | 实现 | 用 git 同步 CodeArts，避免网页上传 | `workdir/code` + `SYNC_CODEARTS.md`；云端 clone/pull+rsync | 进行中 | 命令已写 | 采纳 | — |
| P008 | 2026-11-02 | team | 功能 | 空 Kernel 必 WA，作基线 | 云端 `build.sh -e` | 0.5h | dx 192/192、update 95/192 FAILED | 废弃(空实现) | — |
| P009 | 2026-11-02 | team | 功能 | 按 design_final 实现可过样例 | InferShape+Tiling+Compare/Select Kernel；key 0/1/2 | 本迭代 | 首次云编因 bf16 Cast 失败 | 进行中 | — |
| P010 | 2026-11-02 | team | 环境 | bf16 用 Cast 升精度 | `Cast(bfloat16_t↔float)` | 0.5h | `not support bf16 type cast` → FATBIN 缺 .o | **废弃** | — |
| P011 | 2026-11-02 | team | 功能 | bf16 避开 Cast | 位扩展/RNE 截断逐元素；half/float 仍 Cast | 本迭代 | 编过；样例仍 WA ~97/192 | 采纳(编译) | — |
| P012 | 2026-11-02 | team | 功能 | WA≈半错=Select 极性反 | 交换 Select src0/src1（禁改 LE） | 本迭代 | 换后仍 95/192；见 EXP004 | **废弃(非根因)** | — |
| P013 | 2026-11-02 | team | 协作 | 云日志须落盘公开仓 | 本机 tar→`explorations/logs/EXP20261102_161116`→push | 0.5h | build OK + WA 95/192 原文已归档 | 采纳 | — |
| P014 | 2026-11-02 | team | 功能 | mask 失效→恒选 src1 | 解压日志 + seed42 复算错数；写 EXP004 | 0.5h | 95≡恒正支；97≡恒负支；对上 EXP003 前后 | **采纳(根因)** | — |
| P015 | 2026-11-02 | team | 功能 | 修 Compare/Select mask 通路 | 恢复 Select(g,wg)/(0,xg)；CompareScalar+PipeBarrier；mask 按 256-bit 对齐 | 本迭代 | `EXP_P015_…162823` 仍 WA **97/192**≡恒负支；Compare 仍失效 | 进行中(未过) | — |
| P016 | 2026-11-02 | team | 实现 | Codex 收紧 UB/单缓冲 | `BUFFER_NUM=1`；coef 21/29；usable 80%；同步 design_final | 本迭代 | `EXP_P016_…170730` 仍 WA **97/192**；与 P015 同（不修 mask） | 采纳(UB) / 未过样例 | — |
| P017 | 2026-11-02 | team | 功能 | Select dst/src1 别名踩负支 | 合入 Claude 重写：独立 neg/fg buffer + cmpCount 64 对齐 + DataCopyPad | 本迭代 | `EXP_P017`：**SyncFunc 未声明**，未编过 | 进行中 | — |
| P018 | 2026-11-02 | team | 环境 | 换掉 SyncFunc | `SetFlag`/`WaitFlag` S→V；记 EXP006 | 本迭代 | 已改 workdir；待云端重编 `-e` | 进行中 | — |
| P019 | 2026-11-02 | team | 环境 | P018 实际编译了旧 code 副本 | 加 `verify_codearts_source.sh`，构建前比对 workdir/build kernel SHA256，禁止残留 SyncFunc | 本迭代 | EXP_P019 verify **PASS**；编挂在 bf16 Cast | **采纳(门禁)** | — |
| P020 | 2026-11-02 | team | 环境 | bf16 禁止 Cast（EXP002 再现） | bf16 位扩展/RNE；half 仍 Cast；记 EXP007 | 本迭代 | 已改 kernel；待云端 `-e` | 进行中 | — |
| P021 | 2026-11-02 | team | 交付 | 先交 tip 工程试平台 | 打顶层 `code/` ZIP（beaa00e）；排除 pyc/build | 0.5h | `deliverables/PReluGradUpdate_2329_code_best.zip` SHA256 `A7C800F6…7F973D` | 采纳(试交) | — |

## 性能道路预声明（未到阶段勿开）

| ID | 触发条件 | 默认策略 |
|----|----------|----------|
| P1xx | 功能精度基线采纳后，全核 case < 95% TBE | 标准 tiling/全核等有界优化；触止损写 DEC |
| P1yy | 小 shape <10us 且差 ≤3us | **仿真图 + 分析**，默认不抠微秒 |


## A20261103-PRECISION — 精度深化分析（暂停断点）

2026-11-03：沿云端 P100 完成三视角分析与 CPU 转换探针；主报告见 [精度方案](analysis/AI-precision-plan-20261103.md)，状态见 [恢复断点](analysis/AI-resume-20261103.md)。三组有限值转换差异均0，仅为CPU证据；没有新的NPU/TBE/Judge结果，没有修改算子。用户要求暂停；后续先补三dtype设备精度证据，再考虑C=1/S=1分派。本地旧基线上的追加记录，尚未推送，合并时保留云端新历史。


## A20261103-PRECISION-R2 — 恢复后按 P103 更新方案

2026-11-03：云端main核对22ec6d6，默认P103（47db93e）；当前kernel SHA256与EXP_P103 SOURCE一致。已审6操作/4屏障零缓冲重排，补[16类设备判别与修复规则](analysis/AI-precision-P103-validation-20261103.md)。当前平台仓库记录42.89分、5.83–7.36μs；不再以旧ms状态诊断。V08溢出中点的CPU构造已验证，未做NPU/TBE运行。未改实现；方案完成，设备验证待执行；本地追加记录尚未推送，不能覆盖云端较新历史。


## JadeYara J001（独立座位） — 2026-11-03

| ID | 假设与范围 | 结果 | 下一闸 |
|---|---|---|---|
| J001 | 固定远端22ec6d6/P103复制到本人目录；仅Host将C=1、weights=[1]归为共享权重连续tile；不改其他座位 | 原2项CPU检查与新816组模型PASS；Kernel SHA256与P103一致；新增4项实际Host UT未运行 | Linux/CANN编译、三dtype设备泛化、源码/包绑定同机A/B；未验证加速、不推送 |

依据：[JadeYara说明](../../../people/JadeYara/PReluGradUpdate_2329/README.md)、[开源来源](../../../people/JadeYara/PReluGradUpdate_2329/SOURCES.md)。本地HEAD仍fc85a48，此行基于固定远端22ec6d6；保留原有未提交分析，未用旧本地源码代表P103。

## JadeYara J002 — 2026-11-03：提交1109131正确性诊断（进行中）

用户提供平台结果：2026-11-03 22:53:30，Wrong Answer，测试点1错误100%、3.48us，点2–5 Skipped；尚无提交包身份绑定，耗时不作为有效成绩。本轮只审计J001/P103差分与现有证据，并在JadeYara个人目录增加可运行的针对性诊断；原基线、候选包及其他座位保持原样。无证实根因不改生产Kernel；不上传、不提交、不推送。完成后追加结论。


### J002 本轮结论

只读核实J001 ZIP与workdir逐文件一致，未找到1109131上传包关联；独立审计未发现可证实生产缺陷。补本人experiments/J002参数化三dtype设备runner及30组有限精确诊断；15项CPU故障注入自测、30组/3,609,882输出元素参考对照PASS，缺失设备输出正确返回失败。C++/CANN/NPU未运行，待实际上传包hash、失败输入与设备A/B。保留J001/P103不变，未生成新提交包、不上传不推送。见[调查与执行方法](../../../people/JadeYara/PReluGradUpdate_2329/experiments/J002/README.md)。


## J002-910C — 2026-11-04 实机验证开始

用户授权恢复2329真实设备编译、泛化精度及精度通过后的性能测试；不提交竞赛、不推送、不改网络安全配置。唯一目标：在自己的独立远端目录验证固定P103/J001，保留源包，先查架构兼容和正确性。实读CANN 9.0.0/aarch64，ACL SoC=Ascend910_9362，可见2逻辑设备，npu-smi健康OK且无设备进程。原build.sh/OpDef固定ascend910b，先尝试明确910C目标构建，若需兼容补丁另存副本/差分，不覆写原包。远端计划目录/home/developer/2329-JadeYara-20261104-1812；原始日志暂存C:/Users/lin/Documents/Codex/2026-11-04/task，收尾归档。1109131上传包身份仍未知，不认定为J001。当前未有新编译/精度/性能结果。


## J003 — 2026-11-04 BF16 向量转换实验

P103/J001 实机各54组精度通过；msprof 两轮显示 J001 BF16 [600000,1,1] 中位数673.406/672.746us，AIV scalar ratio=0.995。当前 kernel 逐元素 GetValue/SetValue 转换是明确瓶颈。新路线只在 JadeYara/experiments/J003 的 J001 副本改 BF16 输入/输出为向量 Cast，公式/tiling 不变，FP32/FP16 不变。先 ascend910_93 编译、54组精度，再同机测量；生产workdir和原包保留。未提交/未push。


## J002-910C 结果 — 2026-11-04

P103/J001 ascend910_93三dtype独立编译成功，真实设备各54/54精度通过；两轮msprof每组丢弃5次预热后20样本。FP32 [600000,1,1] P103约5526us→J001 9.17/9.60us；BF16约8633us→673.406/672.746us。12次性能采样输出均正确。BF16 AIV Scalar ratio=0.995，推动独立J003向量Cast实验。完整日志、输入输出、构建溯源和profiler原始数据已下载到 explorations/logs/EXP_J002_910C_20261104。原证据ZIP SHA256=791daa2846ebb45f27f284d8ae0522129eaa1233ce430332b43283f312b7c90f。未提交/未push；1109131包仍未知；这些不是平台五点成绩。


## J003 结果 — 2026-11-04

三dtype编译成功，54/54设备精度通过；BF16 [600000,1,1]约673us→9.28–10.07us，[2,3,65]约6.03–6.31us→4.94–4.99us。8次profile最终输出全部正确。只改BF16转换为向量Cast，保留为个人J003候选，不覆盖J001。完整日志已下载至explorations/logs/EXP_J003_910C_20261104，详情见个人J002/PROGRESS-910C-20261104.md。未push/未提交，公开五点及1109131版本仍待绑定。


## J004 — S=1逐通道权重向量化

扩展J001/J003精度各24/24通过（累计78/78）。新增性能采样定位S=1/C>1路径的逐元素tile循环开销。J004只针对该路径：连续搬运同一batch内权重向量，使用Mul；其它路径保持J003。单独weights队列，Host UB预算增加sizeof(T)，tile在C边界截断。只改个人J004副本，编译后使用既有78组及额外S=1/UB边界设备精度、同机A/B。保留J003候选，无提交/无push。


## J004 结果 — 2026-11-04

J001/J003扩展精度各24/24通过，新增S=1性能瓶颈已复现；J004三dtype编译、96/96设备精度通过，16次两轮性能采样输出全对。S=1用例约26–32倍收益，原路径BF16对照回退0.28–0.50us，因此J004仅作专项候选，不替换J003通用候选。精确延迟见 people/JadeYara/PReluGradUpdate_2329/experiments/J004/RESULTS.md，完整云端日志已下载至explorations/logs/EXP_J004_910C_20261104。J003原包未变，未提交/未push。


## J005 — 编译期隔离S=1权重向量路径

J004 S=1收益已复现，但BF16 [2,3,65]普通路径两轮回退0.50/0.28us。J005只把vectorWeights设为模板bool，kernel入口按tiling选择true/false实例，让普通路径内部的新增条件由编译期消除。Host、UB预算、公式、同步、队列动作均保持J004语义。独立目录保留J003/J004，做96组精度及同机三版交替性能。若无重复收益不取代已有候选。


## J006 — Host tiling key 完整编译期分派

J00596/96精度通过，普通BF16两轮优于J003，但FP16一轮持平、另一轮慢0.37us，尚不能确认通用回退消除。J006只把剩余入口runtime判断移至Host tiling key：0/1/2为普通FP32/FP16/BF16，4/5/6为S=1向量权重；每个kernel入口constexpr选择模板实例。算法、Host分块与UB预算仍保持J005。原J003/J004/J005全部保留。重新编译6key及设备精度，三版同机对照。


## J005 结果 — 2026-11-04

设备精度96/96通过，同机两轮原始日志全部归档。具体测量与选择见 people/JadeYara/PReluGradUpdate_2329/experiments/J005/RESULTS.md；未提交/未push。


## J006 结果 — 2026-11-04

设备精度45/45通过，同机两轮原始日志全部归档。具体测量与选择见 people/JadeYara/PReluGradUpdate_2329/experiments/J006/RESULTS.md；未提交/未push。


## J007 — J003多tile输入预取

依据此前PipeUtilization与顺序CopyIn→Compute→CopyOut代码，尝试在计算当前tile前发起下一tile输入DMA。只将grads/features队列缓冲数改为2；输出仍1，原算术和同步保留。Host UB系数FP32从25增至33，低精度33增至37。按实际currentNum和通道剩余长度推进，保留J003/J004原候选。做三dtype完整96组精度、J003同机交替性能及小输入对照，记录收益与退化，不提交/不push。


## J007 结果 — 2026-11-04

96/96设备精度通过，16次两轮性能采样输出全对。完整结果见 people/JadeYara/PReluGradUpdate_2329/experiments/J007/RESULTS.md，精简证据已下载至explorations/logs/EXP_J007_910C_20261104；完整二进制归档留在云端，SCP下载中断。

结论：当前J007输入预取/双缓冲方案不采用，J003通用候选与J004 S=1专项候选保持不变。

大FP32 [600000,1,1] 两轮分别慢0.300/0.730μs；普通FP16 [17,3,1025] 第一轮慢0.460μs、第二轮快0.741μs；大BF16第一轮快0.340μs、第二轮慢0.670μs，收益方向不稳定。小BF16对照两轮慢约0.120/0.041μs。没有把某轮最低值当作已实现加速。

profiling也未给出一致改善：大BF16第一轮MTE2 ratio由0.3925降至0.3485、耗时小幅下降；第二轮J007为0.4245、J003为0.406，耗时反而增加。流水线ratio可重叠，不能简单相加；这些数据只支持当前方案没有稳定总耗时收益，不足以单独确定某条指令的因果成本。

该实验同时增加输入缓冲和提前搬运，并按实际新增缓冲提高UB系数；更小的maxTile及队列调度成本可能抵消重叠收益，这是待进一步隔离的解释。后续更有价值的候选是先复用中间缓冲、降低UB占用，再重新评估tile/预取配置；本轮未实施该候选，也未承诺4–5μs。

独立续跑作业退出码0，所有设备任务已结束。控制会话中断后的已完成72组没有重跑，仅续跑剩余24组和性能。未提交平台、未push、未修改网络配置。



## J008 — 2026-11-04 中间缓冲复用实验

J007预取没有稳定收益。本实验基于J003，仅在features最后一次Compare读取后复用其存储计算weight×grads，移除neg缓冲及4字节/元素UB占用，保留四道Vector屏障。不引入Select原地写。FP32预算25→21，低精度33→29。运行三dtype边界精度和同机交替A/B；保留J003/J004/J007。


## J008 结果 — 2026-11-04

96/96设备精度通过，20次性能输出全对。完整报告：people/JadeYara/PReluGradUpdate_2329/experiments/J008/RESULTS.md；完整证据：explorations/logs/EXP_J008_910C_20261104。

J008保留为已测FP32大输入的优化候选，暂不替代J003通用候选；J004仍用于已有S=1专项。J008的收益范围目前只落实到本轮[600000,1,1] FP32、w=1这一输入，不能扩展成所有FP32或所有大输入结论。

该FP32输入两轮分别从9.2595→8.6600μs、9.2200→8.7090μs，降低0.5995/0.5110μs，约6.47%/5.54%。正向和反向顺序均有收益。MTE2活跃比例同时从0.4820→0.4425、0.4405→0.3995；这些比例只提供伴随证据，不足以证明总时长降低完全来自搬运或tile次数。此次修改同时减少UB占用并改变Muls所在阶段，尚未做两因素分离实验。

其他输入没有两轮一致收益：FP16通道输入第一轮慢0.7700μs、第二轮快0.6795μs；大BF16第一轮快0.8800μs、第二轮慢0.1195μs；大FP16第一轮慢0.4900μs、第二轮快0.3800μs；小BF16第一轮慢0.0395μs、第二轮快0.4595μs。保留全部采样，不用某轮最低值宣称普遍加速。

下一步优先扩展FP32大输入及UB分块阈值附近的同机对照，验证收益范围；若要明确原因，可单独固定原tile预算分离缓冲复用与tile变大两项影响。上述后续尚未实施。当前大输入仍约8.7μs，没有达到4–5μs；小输入计时不代表竞赛五点成绩。

本轮作业正常结束，退出码0；96/96设备精度及20次性能最终输出全部通过。结束后npu-smi显示设备健康、无运行中的NPU进程。J003/J004/J007原候选和证据均保留，未提交竞赛、未push、未修改网络或安全设置。



## J009 — 2026-11-04 权重索引缓存

J004已覆盖S=1权重向量化；J008在大FP32有5.54–6.47%收益，但Scalar/MTE2仍占较大比例。现有代码每tile重复GetValue、half/bf16解码及SyncSV；J009基于J008，仅当weightIndex变化时读取、解码和同步。初始索引-1保证首tile加载，不修改Vector屏障、算术和Host tiling。全96组精度、与J003/J008同机交替对照，同时监测通道变化路径额外比较的回退。保留所有既有候选，不重做J005–J007路线。


## J009结果 — 2026-11-04

102/102设备精度通过，36次性能输出全对。报告：people/JadeYara/PReluGradUpdate_2329/experiments/J009/RESULTS.md；完整证据：explorations/logs/EXP_J009_910C_20261104。

J009不作为通用候选采用，保留J003通用、J004 S=1专项及J008已有候选。此次缓存没有为目标多tile大输入带来明确、可重复的净收益。

相对直接父版本J008：大FP32第一轮慢0.6895μs、第二轮快0.0895μs，收益方向反转；大BF16分别慢0.0500/快0.0200μs；大FP16两轮仅快0.0100/0.0105μs，处于本轮采样波动范围，不能作为有效加速。FP16通道输入两轮慢0.7200/0.0300μs，小标量FP32两轮慢0.5100/0.2995μs，存在重复回退。

小BF16 [2,3,65]两轮为5.0300/4.5600μs，比J008快0.0700/0.3700μs。这一输入按分块模型几乎没有同索引复用，不能把差异归因为权重缓存命中；尚不足以推广缓存路线，也未连续两轮达到4–5μs。保留实测点，不用单轮4.56μs代表平台五点达标。

三版本同轮对照进一步支持J008大输入方向：本轮大FP32相对J003两轮均更快；FP16/BF16大输入也均更快，但此前实验中存在方向反转，尚不能由本轮扩大通用采用范围。J009增加索引比较和持久状态，其开销可能抵消省下的读取/同步；这是推测，未做指令级归因。

下一项建议转向小输入的有效核数和块大小测量：当前小输入分散到较多核、Scalar活跃比例高，而Vector比例较低。先验证不同核数的同机延迟，继续保持32B块边界和输出不重叠；这是待实施候选，不声称减少核数必然更快。停止扩大本次权重缓存方案。

本轮102/102设备精度、36次性能最终输出全部通过；主作业和新增缓存边界测试均退出0。没有重跑未知状态任务。J003/J004/J008及J005–J007证据均保留；未提交竞赛、未push、未修改网络或安全设置。



## J010 — 2026-11-04 小标量输入分核

J009索引缓存无可采用净收益；小标量9001元素的Vector活跃约4%、Scalar约51–54%、MTE2约49%，并使用38/39个核。选择Host分核而不再增加kernel分支：基于J003，wlen=1目标核数min(硬件核数,ceil(total/1024))，保留32B块对齐。wlen>1及大输入核数策略不变。验证基础96组+18组1024/核心数封顶边界精度，三dtype小输入与原路径控制组交替A/B，并读取msprof实际Block Num。保留所有已有版本。


## J010结果 — 2026-11-04

114/114设备精度通过，28次性能输出全对。报告：people/JadeYara/PReluGradUpdate_2329/experiments/J010/RESULTS.md；证据：explorations/logs/EXP_J010_910C_20261104。

保留J010作为已测[9001,1,1]单权重输入的有收益候选，不作为所有小输入的通用替换。J003/J004/J008保留；本轮只证明一个固定分核策略在具体测试点的效果，未声称1024元素/核是最优参数。

9001元素的三dtype两轮均显著改善：FP32从6.0200/6.8295降到3.5600/3.6400μs；FP16从6.3000/6.3000降到3.4900/3.4700μs；BF16从5.6695/5.6695降到3.1400/3.3700μs。六组同轮比较降低约40.6%–46.7%，中位数均低于4–5μs目标。实际Block Num从FP32的39、FP16/BF16的38降到9。该结论仅适用于上述本地输入，尚未绑定竞赛五点。

极小63元素输入不能直接沿用此策略：FP32从2.6400/2.7600升到4.4495/4.5300μs，BF16从2.3600/2.3400升到4.7100/4.8300μs，实际核数分别8→1和4→1，均重复回退。不据此推广为“所有单核都慢”，因为其他规模尚未做完整性能扫描。

原路径对照中，BF16 [2,3,65]仍为25核，两轮差异+0.1200/+0.0600μs；大FP32仍为40核，第一轮快0.5495μs、第二轮慢0.0100μs。这些对照显示测量有波动，但9001元素两轮、三dtype的2.2995–3.1895μs收益显著大于本轮对照观测到的偏移。

以9001 FP32首轮为例，Scalar ratio从50.65%升至67.90%，总耗时却从6.02降至3.56μs；MTE2 ratio从47.85%降至23.60%。再次说明ratio不是独占耗时。此次有效变化是参与核数，尚未分离多核调度、DMA排队、同步等各自贡献，不能把全部收益归因于某一个流水线比例。

下一步应围绕输入规模测出缩核策略的适用下限，保留已知63元素的原分核，再考虑与J008大输入路径整合。上述边界性能扫描和整合尚未实施，当前不覆盖所有小输入。

114/114设备精度和28次性能最终输出全部通过，主作业退出0；最终npu-smi显示设备健康且没有运行中的NPU进程。完整证据归档校验后保存，未提交竞赛、未push、未修改网络或安全配置。



## J011/J012 — J010规模阈值扫描

用户要求63–9001多点、边界和三dtype泛化，避免单形状硬编码。比较J003、J010、J011最少4核、J012最少8核规则；新规则均只改Host，仍按ceil(total/1024)增长并以硬件核数封顶，保持32B块边界。采用单进程多case批量runner，先用两case pilot验证输出及profiler记录顺序/shape绑定，再做正反顺序批量A/B；每case仍25次调用、丢弃前5次。选择规则后另用未参与选择的规模和多维shape做验证。保留所有现有候选，不提交、不push。


## J011/J012规模扫描结论 — 2026-11-04

选择J011作为已验证小标量权重输入的候选，J012保留记录但不选用。J011目标核数为min(硬件Vector核数,max(4,ceil(totalNum/1024)))，实际usedCores继续按32B块对齐和有效块数计算，可能少于目标。wlen>1路径保持J003规则；kernel与J003相同。

在30个规模×三dtype的90个扫描点中，J011有86点两轮均快于J003至少0.15μs，其余4点接近持平；所有180组配对比较中最大观测回退仅0.01μs。J011批量测量的中位数范围2.15–3.28μs。J012目标下限8通常比J011慢；J010在部分驻留测量中更快，但63/64元素的结果随采样顺序变化，不把它解释成稳定的单一元素数分界。

J011的规则在total≤4096时保持目标4核，超过4096后按1024元素/核增长并以硬件核数封顶；4095/4096/4097、8191/8192/8193等边界均已测。4096是该通用规则的核数增长边界，未宣称是全局最优参数。没有对某个已知shape作相等判断或硬编码9001。

36个留出输入包括33个单权重用例和3个通道权重对照。33个单权重用例两轮都更快；通道对照基本持平。它们包含未用于选规则的97/193/777/1537/3001/5003/7001/8999元素和不同C的多维shape。

原单输入profiler方式的FP32校准也通过：63元素由J003的2.6700/2.6395降至J011的2.2200/2.2000μs；9001元素由6.1895/6.2400降至3.6800/3.8400μs。上述均是20样本中位数，不是最坏时延保证，原始min/max保留在CSV。

本轮支持采用J011小标量路径，保留J003、J004和J008的其他已测用途。尚未把J011与J008大输入kernel或J004 S=1路径合并。性能范围限本次63–9001扫描及列出的留出输入；竞赛五点未绑定，不宣称平台达标。

完整报告：people/JadeYara/PReluGradUpdate_2329/experiments/J011/RESULTS.md；完整证据：explorations/logs/EXP_J011_J012_SCALE_910C_20261104。


## J013 — J011/J004/J008 按路径整合（2026-11-04）

用户授权的新目标：保留原版，将 J011 单权重分核、J004 S=1 通道权重、J008 普通路径 UB 复用整合为新候选。仅个人 experiments/J013 实现；不提交、不推送。普通 kernel 保持 J008，S=1 复制 J004 仅改命名空间/头保护，使用独立 tiling key 0/1/2 和 4/5/6 编译期分派；Host 按各路径实际缓冲计算 UB，wlen=1 使用 J011 通用目标核数规则。未重复引入 J005–J007 的无收益改动。

原生编译已通过，240 个既有回归输入和 33 个新留出输入均精确通过。54 个 shape/dtype 输入 × 五版本 × 两轮完成 540 组 profiler 计时；每组 25 次、丢前 5 次。极小 S=1 [3,5,1] 第二轮有约 0.9–1.1 us 回退、普通路径差值随顺序反向，正在固定 case 顺序做四轮定点交替，完成后按结果定是否采用。当前不可仅凭单轮快慢定收益。


## J013 整合完成（2026-11-04）

J011 单权重分核 + J004 S=1 独立 key + J008 普通路径 UB 复用已整合，原版保留。273/273 精度输入通过；五版本两轮 540 组和三路径四轮复测 160 组输出全部通过。S=1 极小输入第二轮约 1 us 回退在固定顺序复测中未复现，普通路径有顺序敏感性，未发现可持续回退；J013 保留为整合候选，不宣称相对专项版额外独立加速。9001 单权重已在本地达到目标区间内，大 S=1/600k/跨 tile 通道仍超目标。详情 people/JadeYara/PReluGradUpdate_2329/experiments/J013/RESULTS.md；完整证据 EXP_J013_INTEGRATION_910C_20261104，已校验 SHA/CRC/二进制索引。无在途设备作业、未提交、未推送。


## J014 — S=1 对齐通道合并计算 tile（2026-11-04）

以 J013 为基线，S=1 [127,256,1] FP32 profiling MTE2 61.25%、Scalar 52%，每核约 4–5 段通道循环。尝试仅在通道字节数 32B 对齐且 tileLen>C 时，把权重按 C 边界分段搬入同一队列后统一 Compute/CopyOut，减少 grads/features/output 搬运调用及 Vector 同步重复。奇数通道保留原路径；600k 普通路径仅作对照，本轮不改。普通路径 FP32 600k MTE2 42.25%、Scalar 37.75%、Vector 26.25%，占比重叠，当前不足以证明已排除的双缓冲值得重试。计划完整 J013 273 回归 + 新对齐边界及跨 tile 输入，三 dtype、同机四轮交替；结果待测。


## J015 — J014 权重周期在 UB 内复用（2026-11-04）

J014 369 精度输入通过，但四轮 [1001,512,1] 低精度回退约 6–7%，不直接采用。J015 仅改 J014 权重展开：GM 读取一个旋转后的 C 周期（最多两个分段），DeQue 保证 MTE2→V 后，用 UB→UB DataCopy 从第一周期复制其余位置，PIPE_V 屏障后再 Cast/Mul。C 字节对齐条件、容量、算术和其他同步不变；尾段复制数按 32B 对齐，目标不超过原 alignedCount。CANN 9.0 本机 DataCopy LocalTensor 重载与 UB 路径已核查；官方 API pipeline 表确认 UB→UB 属 PIPE_V。保留 J014 全部结果，计划完整 369 输入回归加新随机留出，再 J013/J014/J015 四轮对照。


## J014/J015 完成：S=1 权重周期复用（2026-11-04）

J014 合并 tile 精度369通过，但仍重复读GM权重，收益不稳且有回退，未采用。J015 仅进一步将同一权重周期在UB展开，精度399通过，三版本四轮1476组及位置复核72组输出全通过；63个触发输入四轮平均全部快于J013，54个每轮均快≥0.15us。[127,256,1]约提速28–34%，[1001,512,1]约68–75%。600k未优化、仍约8–9us。极小未触发S=1有约0.045–0.175us的位置复核平均开销，因此J013保持综合基线，J015保留为对齐S=1专项源码候选；下一步独立编译分派再评估600k tile预算。报告 people/JadeYara/PReluGradUpdate_2329/experiments/J015/RESULTS.md；主/补完整证据已下载校验于 EXP_J014_J015_S1_REUSE_910C_20261104。无在途作业、未提交、未push。


## J016 — 补测仓库建议的整核权重驻留（2026-11-04）

用户要求逐项核对仓库建议。SUMMARY 的模式 R、design_final §3.3.3（171–179行）明确“整核只展开一次”；J013 仅连续向量搬运，J015 只做到每tile周期复用，未完整落实。这一差距现在补测：J016 基于J015，只在对齐S=1、多tile且tileLen向下对齐C不增加ceil(blockFactor/tileLen)时，首tile构造权重后保持队列张量到本核结束，其余tile复用，rowsPerTile记录真实完整周期数。算术/UB容量不变；phase通过tileLen为C倍数保持。原J013/J015保留，另测完整精度、驻留与不适用边界、同机A/B，不提前认定有效。


## J017 — 复核最新上游 P105 的提前权重准备（2026-11-04）

本轮实时核对协作仓main=e62693c8，较旧固定参考22ec6d6已有P104拒绝及P105（2e378eb）样例通过记录；P105尚无性能结论。J017基于J013，只将普通kernel的标量权重index/GetValue/decode块移到Compute入口、输入DeQue之前，保留SyncSV在原位置；不改Host、UB、S=1或数值。此处是把P105思路应用到已向量Cast/复用buffer的J013，不声称等同完整P105包。与J013同机独立A/B，等待J016设备作业完成后才启动。


## J016 失败定位 / J018 单条件修正（2026-11-04）

J016既有399通过、新留出32/36通过；失败仅FP16/BF16 [127,4096,1]/[129,4096,1] 的dx，update全对。Host将tileLen下调为C后，mergeWeights旧条件tileLen>C为false，Process仍在C边界截短，导致缓存权重相位不再固定。J018仅将mergeWeights启用条件扩成tileLen>C或residentWeights，保证驻留时不重新通道截断，其他代码与J016一致。失败源码/输出全保留，J016未跑性能；J018重跑全部435及新增可解释周期边界，然后才计时。


## 仓库建议复核阶段结论

已重新核main=e62693c8/P105，逐项对照见 people/JadeYara/PReluGradUpdate_2329/UPSTREAM-AUDIT-20261104.md。明确此前未落实整核驻留；J016新留出4例失败，定位tileLen==C的相位问题，J018修正版准备未运行。最新P105思路J017设备435精度/312计时组通过但无稳定性能收益，不采用。J014/J015完整归档已完成；J016/J017完整云日志尚待下载，当前无在途设备作业。按用户要求先报告，不将待执行写为完成。


## J018 设备精度通过，四版本计时进行中

2026-11-04 22:32 UTC+8：J018 已在既有 910C/CANN 9.0 环境构建成功。J016 原失败 shape 的三 dtype 6 个案例复现 2 PASS/4 FAIL；J018 同输入 6/6 PASS。J018 全量既有 435/435 与新增 27/27 PASS，共 462 个独立输入；新增含全负 features、周期 31 通道权重的可解释相位数据。四版本性能集共 69 输入，J016 精度资格检查 61/69 PASS，失败 8 例不计性能。J013/J015/J018 使用完全相同 69 输入，J016 只在合格子集计时。已开始四轮轮换顺序，每组 25 次调用、舍弃前 5 次、取后 20 次中位数；当前不提前下性能结论。

J018 唯一修正为 mergeWeights 条件允许 residentWeights；source ZIP SHA-256 b32e33df1cb55e09bf3fc0a0be6d44c03d075282c9ce102e306d238d72180932。原 J013/J015/J016 保留。


## J018 完成：驻留相位修复与完整归档（2026-11-04）

J018 单条件修复构建通过，6个失败边界的三dtype复核全部正确，435既有+27新增=462独立精度全部通过。J016同输入复现原4失败，新增相位再发现4失败，8项排除性能。J013/J015/J018各69输入、J016合格61输入，四轮共1072组/26800 kernel记录，输出全通过。Host规则计算28个驻留输入平均全部改善，27个每轮至少快0.15us；BF16 [1201,512,1] 16.384→10.720us，[127,4096,1] 10.880→9.517us。非驻留路径回退：BF16 [2001,512,1] 23.119→25.679us，小S=1也退化，故J018仅保留专项候选，不替换J013/J015。下一步候选是独立编译隔离驻留路径，本轮未实施。

J016/J017/J018完整云端证据已下载，SHA/CRC/全部二进制索引验证通过：J016-J018-910C-evidence.zip 354043635 bytes，SHA 2187aaced95a13649de82f014af9b5f0f672eb18cb92d1edccaf1eaf83b507ef。报告 people/JadeYara/PReluGradUpdate_2329/experiments/J018/RESULTS.md；归档 tasks/active/PReluGradUpdate_2329/explorations/logs/EXP_J016_J018_RESIDENT_910C_20261104。无在途NPU作业，无提交或push。


## J019 — 独立编译驻留与回退路径（2026-11-04）

以已测 J018 为基础，仅分离编译路径：普通路径保留 J013/J008，未合并的 S=1 使用原 J013/J004，小通道合并但不驻留使用原 J015，驻留使用独立 J018 类并把 resident/vector/merge 标志编译期固定为 true。Host 仍按通道对齐、UB 容量、向下对齐不增加循环数的通用规则分派；没有 shape 白名单。原三候选保留。计划完整462精度+新增容量阈值附近三dtype留出，再四版本相同输入四轮交替，检查驻留收益保留和小输入/非驻留回退消除。


## J020 — 仅独立分派，撤销 J019 驻留标志常量化

J019 已通过498精度和1680组输出，但驻留44例中28例每轮比J018慢至少0.15us。J019同时分派与编译期常量化，需拆因果；J020只将 resident header 恢复为原J018实现（仅命名空间/guard变化），其余Host与key及fallback与J019完全一致。先498回归，再与J013/J015/J018/J019同机四轮对照，优先覆盖已退化路径和固定位置小输入。


## J019/J020 独立编译实验完成

两版各498精度通过，1680+780+420=2880组输出全通过。完整对照见 people/JadeYara/PReluGradUpdate_2329/experiments/J020/RESULTS.md。

## 本轮结论与采用状态

**独立分派实现与精度通过，但尚未全面隔离小输入回退；J019/J020 不作为新的全量默认候选。** J013 保持综合基线，J015 与 J018 的原专项候选继续保留。本轮没有声称达到未知榜单的 4–5 μs 目标。

固定位置复核给出了更具体的边界：

- 大输入之前的 [3,5,1]，J020 相对 J013 的平均差值为 FP32 +0.028、FP16 +0.005、BF16 -0.025 μs，接近原水平。
- 驻留大输入之后，同一小输入 FP16 为 2.950→3.550 μs，BF16 为 2.970→3.592 μs；四轮差值分别约 +0.57～+0.63、+0.59～+0.64 μs。
- 非驻留大输入之后，FP16 小输入 2.948→3.347 μs，BF16 2.977→3.362 μs；四轮分别 +0.34～+0.43、+0.37～+0.40 μs。J019 也有相同现象。因此这不是只在长序列里出现的一次异常。
- 大输入的权重驻留收益保留：固定位置 BF16 [1201,512,1] 为 J015 17.637、J018 10.950、J019 10.802、J020 10.732 μs；FP16 同 shape 为 18.247、10.775、11.220、10.942 μs。新分派没有证明所有 dtype 都优于 J018。
- 非驻留 [2001,512,1] 的固定位置 J020 与 J015 相比：FP32 +0.056、FP16 -0.638、BF16 -0.185 μs；原 J018 大幅回退在此短批没有复现，但不能据此抹去早期测试差异。
- 600k 普通路径 J020 相对 J013 的固定位置平均差约 +0.075/+0.000/-0.045 μs（FP32/FP16/BF16），没有新算法收益。

编译期常量化不是普遍收益：J019 在 [40001,16,1] 的低精度路径有额外改善，但其他驻留路径的差值随 campaign 变化，J020 恢复原实现后也没有解决所有小输入后继开销。当前证据不能单独把问题归因于指令缓存、频率或某个同步事件。

固定位置复核控制了输入顺序和位置，但前序大输入在各版本中的实际执行时长不同（例如 J013 数十 μs、优化版十余 μs），没有把时钟、缓存或前序执行状态固定成相同值。因此低精度小输入的差异是该调用序列下的实测现象，不足以证明小输入核代码本身增加了同等开销。若要求跨序列都无回退，此验收条件尚未满足。

进一步读取已有 profiling：BF16 tiny_after_resident 的 J013→J020，Task Duration 为 2.970→3.592 μs，aiv_time 为 2.434→3.066 μs；FP16 对应为 Task 2.950→3.550 μs、aiv_time 2.413→2.998 μs。额外差异主要伴随设备 AIV 执行区间增长，下一步应从该区间查起，不能将其归于 SSH 或本机网络延迟。Scalar/Vector/MTE2 活跃比例仍相互重叠，不足以定位独占等待原因。原始聚合见 position-metrics.json。

后续应先定位“低精度小输入紧跟大输入”的稳定额外开销，结合入口反汇编、代码布局与流水线事件做一个有针对性的实验；不继续同时修改分派和数值路径。本轮没有进一步修改代码。


归档 tasks/active/PReluGradUpdate_2329/explorations/logs/EXP_J019_J020_DISPATCH_910C_20261104；增量归档与指定J016-J018完整归档共同校验通过，可恢复全部证据。无在途NPU作业、无提交或push。


## J021 — 大输入后小输入回退的控制实验

不改算子。已有原始counter显示低精度tiny_after_resident J013 I-cache miss rate=0，J020约0.038/0.039，AIV cycles同步增加；FP32对照miss=0。cycles/aiv_time为聚合换算，不能当独立实时频率。旧驱动每case分配/释放设备内存，需隔离地址与顺序。J021仅新建诊断驱动，对J013/J020做fresh/reuse固定地址、前序resident/merged/scalar、25或250次前序、10ms间隔、同地址输入重写；tiny连续250次分早/中/晚窗口统计，四轮交替。已有dtype/输入/golden复用，无安装、无频率设置或安全修改。


## J022 — 低精度 legacy S=1 的指令预取实验

J021 已复现目标小算子预热消除回退，而无关大算子预热不能消除；ICache miss 同步变化。仅在 J020 的 FP16/BF16 legacy S=1 入口增加 ICachePreLoad(4)，4×2KB=8KB，现有 AIV 16KB 上限内；其余 Host/数学/同步和三条路径不变。安装 CANN 的 dav_c220 wrapper 支持 preload，API 文档 https://asc.gitcode.com/api/SIMD-API/basic_api/cache_control/ICachePreLoad_ISASI.html 。先构建和498精度，再同机冷/热及泛化交替对照；若无重复收益则保留 J020 原版。J021 replay 独立测量，不与普通 profiler 混用。


## J021/J022 完成：指令缓存顺序控制与预取实验

J021 ordinary profiler 共123800 kernel记录、704组输出；replay32组（96组最终输出）分开口径，目标预热而非无关大算子预热消除小输入回退。J022原生构建、498精度、720组冷态/通用+54组热态输出全部通过。## 结论与采用范围

**保留 J022 作为冷态低精度 S=1 专项候选，不替换 J013/J020 等现有综合候选。**

- 固定位置的 `[3,5,1]`：FP16 在 resident 前后/merged 后，J020→J022 分别为 3.887→3.183、3.493→3.140、3.370→3.115 μs；BF16 为 4.062→3.205、3.560→3.150、3.348→3.115 μs。这六个条件的四轮均改善。ICache miss 没有全部归零；BF16 merged 后的 miss 甚至略升，说明预取降低等待并不等价于命中率必然升高。
- 热态第101–250次窗口：初始/大输入后 tiny 的 J022−J020 为 FP16 +0.005/+0.005 μs，BF16 −0.010/−0.020 μs，FP32 −0.015/0.000 μs；本次两轮未见明显持续热态代价。第6–25次过渡窗口仍依赖上下文，例如本轮 FP16 J020 在大输入后为3.240 μs，不应把“预热250次”当成之后永不发生ICache回退的保证。
- 通用39输入序列仍有明确回退：BF16 `[1001,512,1]` 平均 +0.962 μs，四轮为 +1.359/+0.379/+0.190/+1.919；BF16 `[2001,512,1]` 平均 +1.620 μs，四轮均为正；FP32 `[2001,512,1]` 平均 +0.703 μs，四轮均为正。相同 `[2001,512,1]` 在固定位置序列中没有一致回退，结果依赖执行上下文。这些回退使本轮候选不适合综合替换。预取不在这些模板上执行，代码布局和顺序效应仍需隔离。
- 600k 标量权重输入没有可重复的改善；本轮不宣称完成大输入4–5 μs目标，更不把本地未知对应关系的形状当作榜单五点。

后续先固定执行顺序与热态，复核上述三个退化的大形状；二进制布局和上游 split-C 权重复用分别作为后续候选。本轮已完成定位、最小候选、完整精度和冷热对照，所有候选与回退证据保留。

报告 people/JadeYara/PReluGradUpdate_2329/experiments/J021/RESULTS.md 与 J022/RESULTS.md，完整云端证据已取回并哈希核验。没有push或竞赛提交。


## J023 — 按编译实际 dtype 隔离模板函数体

J022预取只在FP16/BF16 legacy S=1执行，大形状回退路径没有执行该指令；已知其布局位置/字节哈希改变。当前每个输入dtype二进制包含全部三dtype共12个完整模板，原生编译命令定义DTYPE_GRADS=float/half/bfloat16_t。J023以J022为基线，在三个if constexpr分支增加std::is_same<DTYPE_GRADS,T>::value，裁掉当前二进制不使用的另外两种dtype函数体；Host/key/预取与数学不变。先检查三个ELF的实际体积/函数体，再498精度与J020/J022/J023冷热交替。保留J013/J018/J022。当前目标是减少代码布局耦合，不将体积缩小预设为延迟收益。


## J023 结案：dtype编译裁剪与大输入热态控制

代码段65,036字节缩至21,260/23,860/23,100，498精度、720短预热组+216大输入连续热态组全通过，72,000条profile。大输入逐轮波动主要伴随MTE2/MTE3，Vector固定；J023相对J022部分缓解但未全面恢复J020，BF16单元素新增约0.150us回退，保留候选不替换综合版本。详细表、原始完整归档及SHA已落盘至EXP_J023_DTYPE_910C_20261105，未提交/未push。


## J024 — S=1 split-C / batch 分核，通道块权重驻留

目标：仅覆盖J018不能完整C驻留的C>maxTile路径。基于J023，新增split-C模板：C分成ceil(C/maxTile)个平衡的64元素对齐块，每通道块分hardwareCoreNum/cTiles组batch，组内遍历batch并持有权重队列tensor；沿用现有CopyIn/Compute/CopyOut数学与同步。仅S=1向量权重、C按32B对齐、cTiles<=硬件核数且每batch组至少2行时启用，保持近满核和权重摊销。tile预算不增加，旧路径保留。先498原精度及新大C/分核/尾块边界三dtype，再J018/J023/J024同机冷热和PipeUtilization对照；上游只借鉴调度概念，许可与commit见J021文档，不直接移植950/arch35实现。


## J024 结案：split-C 跨 batch 权重复用

582 组精度、1296 组性能输出校验全部通过，64800 条 profile。热态 [127,8192,1]：FP16 J023 17.700→J024 12.940 us，BF16 17.785→13.072，FP32 14.180→11.895，四轮均改善；低精度 MTE2 约 8.8→3.3 us。[31,32768,1] 低精度四轮回退 0.7–0.9 us，保留专项候选，不替换全局。初次计划重名失败已修复，不属于设备精度失败；原始日志保留。完整报告和校验归档见 explorations/EXP_J024_SPLITC_910C_20261105.md。未提交/未 push。


## J025 — split-C 泛化路由扫描

延续用户授权，扫描20/32/40核预算与完整/半容量通道tile；仅在独立J025P测量分支用进程环境选择参数，数学和同步继承J024。基于训练集的规模归一化特征选小型路由规则，冻结后编译不含环境探针的J025，使用未参与选择的N/C形状、三dtype及原边界精度、同机交替A/B验证。保留J023/J024，不按具体shape特判，不提交/push。训练与留出清单在测量前固定。


## J026 — J025 留出失败后的窄范围规则

J025三条件训练树在留出[57,18432,1]等形状仍回退，保留失败候选，不继续调参复用同一留出结论。J026回到J024，Host只增加两项几何限制：通道最多3个UB块，batch至少2倍硬件核数（本机80），仍用40核预算/原tile。含义为通道向量只需少量块、连续基线每核至少跨两次完整通道周期。新冻结的独立留出覆盖N=79/80/81、C块数边界与新C；旧留出转为已知验证集。规则仍是经验候选，成功与否以新留出实测为准。


## J025/J026 分核路由扫描结案

J025P完成26训练形状三dtype、20/32/40核及双档tile扫描；J025三条件树675精度通过但独立留出性能失败，保留。J026两条件窄范围708精度通过，11个新留出形状与已知形状/控制完成四轮150次对照。总4518组性能输出校验、330300条profile。结论与逐形状数据见explorations/EXP_J025_J026_ROUTING_910C_20261105.md，完整云端归档及五个依赖SHA均已验证；不提交/push。
## 结论与可用范围

J026 的窄范围规则通过本轮独立留出验证，可保留为目前更稳的有条件 split-C 候选。33个新留出dtype/shape组合中，17个四轮均快于J023、0个四轮均慢；其中按规则启用split-C的18个组合，17个四轮均改善，FP32[81,9216,1]一轮+0.230 μs、其余三轮改善。15个回退到原路径的组合与12个控制组合仍存在轮次变号，不能据此声称完全无波动。没有将J026替换所有旧候选或提交竞赛。

新形状[147,6656,1]：FP16 19.322→13.160 μs（约31.9%下降），BF16 19.404→13.292（约31.5%），FP32 14.640→11.730（约19.9%）；三dtype四轮均改善。[103,9728,1]分别15.870→12.510、15.970→12.737、12.842→11.645 μs（FP16/BF16/FP32）。已知[127,8192,1]保持原split-C收益，J026与J024差值约−0.050/+0.020/−0.002 μs（FP16/BF16/FP32），不把小差值当新优化收益。

原回退[31,32768,1]在J026回到原路径，FP16/BF16较J024平均快0.855/0.808 μs；相对J023差值逐轮变号，已不再出现本轮四轮一致回退。窄规则也会放弃实际可得收益，例如低精度[97,12880,1]相对J024慢约2.0–2.2 μs，因此不是所有shape最快的统一版本。J025扫描与失败树、J024宽范围候选均保留供后续对照。

本轮支持的选择规则是UB容量、通道块数和batch/核数比例，不是具体N/C查表；但阈值仍属本机测量得到的经验范围，只在本次40核910C/CANN9环境验证。后续若扩大范围，应另设新留出集，优先检查4块及以上通道的负载/搬运平衡，不能把本轮已知样本再次当独立泛化证据。此次大输入仍约10–22 μs，未达到4–5 μs，也不能换算未知榜单输入成绩。



## J027 — 固定J026 tiling，缓存低精度权重的FP32转换

目标限定为13us大输入的代价核算与一个可执行候选。J026 split-C仍每batch Cast同一权重，改为首batch转换到已有neg缓冲并驻留；Compare读取features完成后，复用featuresFloat保存weight*grad（沿用J008已验证的生命周期思路）。Host、tile、核数、UB分配不变，FP32数学路径不变。不重复J007双缓冲扫描。上游hicann/ops-nn b5f577d6 CopyWeightTile/ProcessChannelNcSplitCWeightReuse将BF16权重转换移出batch循环，许可与原文已归档J021；本次只借鉴思路。另设J027IO诊断版本，仅保留同tiling搬运/队列并输出grads/features，用于观察相同结构搬运成本，不是合法算子候选或严格硬件下界。先708原精度，再63数据集三dtype四轮J026/J027；IO仅3形状×3dtype。主形状两种msprof op replay模式核实L2影响。全部保留，不提交/push。


## J027 代价核算与缓存转换实验结案

708精度通过，93000条普通profile及24次replay均完成，对应输出全部正确（IO版本单独按identity校验）。Vector转换成本下降但首轮端到端无稳定改善；FP32函数字节完全相同仍有计时差，已补A/B/A。完整结论、IO条件下界、L2命中率与保留/淘汰决定见explorations/EXP_J027_COST_910C_20261105.md及J027/RESULTS.md。原始归档与六个依赖SHA已核验，不提交/push。
## 结论与下一步

本轮保留J026为基线，J027作为待进一步验证的低精度条件候选；不全局替换。J027不是简单的“无收益”：普通63组协议未见稳定收益，但固定分配顺序的A/B/A协议在目标大输入得到低精度四轮一致改善。两个协议不同，不能挑选最快值当成普适结果。

固定[127,8192,1]、101–250次窗口、四轮中位数均值（μs）：

| dtype | J026 A1 | J027 B | J026 A2 | B减两侧A均值 |
|---|---:|---:|---:|---:|
| FP16 | 12.555 | 11.890 | 12.925 | -0.850 |
| BF16 | 12.767 | 12.190 | 13.145 | -0.766 |
| FP32（函数字节未改变） | 11.555 | 11.410 | 11.917 | -0.326 |

未改的FP32 [600000,1,1]也出现-0.341 μs差值。A2相对A1系统性偏慢约0.16–0.46 μs，说明固定A/B/A顺序仍存在状态或执行位置效应。不能直接用FP32偏差扣除低精度得到“净收益”，但低精度额外改善与Vector减少0.384/0.430 μs相符，支持机制有效；总体泛化收益尚未证实。当前只确认分配顺序固定，不声称不同进程得到相同物理地址。

### 13 μs主要花在哪里

同一[127,8192,1]，两轮msprof op结果：

| dtype | 完整J026：保留L2 / 清L2 | IO诊断：保留L2 / 清L2 |
|---|---:|---:|
| FP16 | 12.920 / 14.700 | 8.240 / 9.750 |
| BF16 | 12.760 / 14.750 | 7.970 / 9.590 |
| FP32 | 11.800 / 14.480 | 8.780 / 11.000 |

保留L2时读命中率约99.997–99.999%、写命中100%；清L2时读约7.14–7.34%、写约0.25–0.49%。这些样本CurrentFreq均1650 MHz。FP32清缓存时MTE2每轮最大核活跃时间为7.564/7.081 μs，Vector为3.864 μs；保留缓存时MTE2为3.307/3.364 μs，Scalar为7.773/7.533 μs。低精度保留缓存时Vector约5.497（FP16）/5.718（BF16）μs，Scalar约6.18–8.01/6.75–7.69 μs。这里各指标取各自最大核，不一定同一核，且有重叠，不能拼成耗时饼图或独占占比。

因此冷缓存大输入明确要关注MTE2；热缓存仍有循环、队列、同步与向量执行成本。移除数学计算的诊断路径仍约8–9 μs，是当前结构代价的直接证据，但它改变了指令混合，不是严格的硬件下界。权重Cast只能减少其中不足1 μs的Vector工作，单靠它不能解释13→5 μs的差距。

### 下一项最有依据的有限实验（本轮未实现）

只对低精度split-C路径尝试**保持原tile大小的输入预取双缓冲**，第二套buffer仅给grads/features，权重与输出队列不变。目标是用下一batch的MTE2覆盖当前Vector/输出阶段，而非再次扫描核数或缩小tile。先以同一IO诊断验证是否确实改善调度/搬运，再验证真实算子。此前J007改变tile导致循环数增加；本提案控制tile不变，检验的是不同假设。

从当前InitBuffer逐项核算，低精度原35 B/元素，新增两份2 B输入后为39 B/元素；最大tile4288需要167232 B，小于物理UB196608减8192保留后的188416 B，余21184 B。当前80%预算为150732 B，实施时须显式调整该专用路径预算/检查，不能沿用错误的35 B核算。FP32最大tile5184若同样增加输入槽需191808 B，超过188416 B，故不直接推广到FP32。

限定成本：一个真实候选加一个诊断候选，固定3个现有大输入和少量未使用的相邻形状；先诊断四轮交替，如果无一致改善即停止，不做参数网格。若诊断有效，再做三dtype完整708组回归（FP32路径不改）及留出形状、交替性能；预计实现/构建/验证约20–30分钟，设备作业约5–10分钟，归档另计。接受标准是完整输出正确、同协议收益超出A/A状态差异、未改路径无可复现回退。4–5 μs仍是目标，当前没有证据承诺达到。



# J028: fixed-tile low-precision split-C input prefetch

Scope: JadeYara/PReluGradUpdate_2329 only. User authorized this bounded performance experiment after J027. Preserve J026/J027 and all previous evidence; no push/submission/network changes.
J028 derives from J026, not J027, to isolate prefetch. Only low-precision input queues get two slots. Keep tile/core/routing unchanged; explicitly budget 39 B/element against 90% of reserved UB. Prefetch next batch before current Compute, retain single weights/output slots. FP32 keeps its serial schedule.
Gate: first build J028IO and compare with J027IO identity diagnostic, four counterbalanced ABBA/BAAB rounds, 3 existing large shapes plus 1 new adjacent shape, all dtypes. Stop if no consistent low-precision gain beyond control/state variability; no parameter grid. Only if gate passes, build real J028, run 708 regression plus new shapes and alternating A/B. Estimated 20–30 minutes excluding transfer.


# J028固定tile输入预取：停止

四轮ABBA/BAAB、四形状三dtype诊断，192组identity输出通过、48000条计时。八个低精度组合四轮全回退，均值+0.554–0.744 μs。未通过先验诊断门，停止此方向；真实J028仅源码，未设备编译/精度/性能测试。保留J026/J027，不扫描参数。FP32控制也有+0.32–0.34 μs差异，故不将全部回退归于代码。完整依据见../explorations/EXP_J028_PREFETCH_910C_20261105.md；原始归档含初版编译失败和r1成功日志。


# J029: paired input/output queues in split-C

User authorized a bounded investigation of per-batch transfer/queue overhead. Scope: JadeYara/PReluGradUpdate_2329 only; preserve J026/J027/J028, no submission/push/network change.
Evidence: J027/J028 IO diagnostics retain about 4us Scalar; each split-C batch has four independent queue lifecycles (16 Alloc/EnQue/DeQue/Free calls), while both inputs must be ready before Compute and both outputs are produced together. Packing grads/features into one queue tensor and dx/update into another halves queue operations without changing four GM copies, mathematical operations, tile, core count or UB bytes. Non-overlapping regions use aligned tile offset; one queue event orders both copies. This is a synchronization/queue candidate, not a reduction in GM traffic.
Two-batch full-buffer fusion would require 35*2*4096=286720 B at the target low-precision tile, above 188416 B usable UB, so do not shrink tile or start a fusion scan.
One real candidate from J026. Compile, 708 existing precision cases plus new adjacent split-C boundaries, then four ABBA/BAAB rounds on a bounded set of large shapes and generic control. Stop if no stable benefit; no second candidate this turn.


# J029成对队列：停止，未晋升

723组真实算子精度通过；15组合四轮ABBA/BAAB，240组输出、60000条profile。0/15四轮均改善。目标[127,8192,1] Scalar下降0.873–1.113 μs，Task差仅-0.018至+0.011 μs，未转化为稳定延迟收益。停止本轮队列方向，不叠加新改动；J026保留基线、J027保留条件候选。不能把Scalar活跃时间等同于可消除独占耗时。详见../explorations/EXP_J029_PAIRED_QUEUE_910C_20261105.md，完整云端证据已取回并校验。


# J030: observe dependencies, memory bandwidth and profiler limits

No operator source changes. Investigate J026 baseline versus J029 paired-queue binaries on the existing 910C. Local msprof op exposes PipeUtilization/ResourceConflictRatio/Memory/MemoryUB/MemoryDetail/TimelineDetail, but not on-board InstrTimeline/PipeTimeline. Official fixed-commit guide limits actual instruction/pipe timelines to 950; TimelineDetail for A2/A3 is simulated and restricted to supported Python framework scenarios. Current harness is C++ ACLNN; do not call simulation output real hardware dependency timing.
First probe MemoryDetail on one fixed fp16 [127,8192,1] run. If counters work, bounded two-round reversed J026/J029 comparison for fp16/fp32, kernel versus application replay; inspect stall/wait counters, cache hit and traffic, not summed active time. Add bf16 only if needed, not automatically. Existing ordinary task timeline may be inspected for scope, not mistaken for within-kernel instructions. Stop at the smallest tool/data gap if critical dependencies are not observable. No installs, network/security changes, competition submission or push.


## J030：依赖等待取证完成

17次MemoryDetail采集（1探测+16冷/热、两版本、两dtype、两轮对照）输出均正确。源码未改。取得VECTOR IQ满、核内wait、UB冲突及搬运计数；任务timeline不具备核内指令依赖，main_memory字段不能直接作为HBM实流量。完整数字、推断边界和最小数据缺口见explorations/EXP_J030_DEPENDENCY_910C_20261105.md。J026/J027候选状态不变，完整归档已取回校验。


# J031: installed simulator dependency trace

Use existing CANN9 msprof op simulator, Ascend910_9362 maps to dav_2201 with CA libraries present. Probe one FP16 [127,8192,1] invocation using unchanged J026 ACLNN runner and installed binary, then J029 if successful. Parse selected cores; preserve whole-operator tiling. Bound each simulation to 3 minutes. No source optimization, installs or network/security edits. Simulation cycles are separate from J030 hardware time. If ACLNN unsupported, inspect documented config direct-launch entry before deciding whether components are missing.


## J031：模拟器依赖复现完成

+CANN9现有Ascend910_9362/dav_2201模拟库可用。未改算子，原J026/J029 ACLNN runner各单次FP16 [127,8192,1]仿真成功且输出匹配golden；原40 block与4096 tile通过dump复核。获得PC/输入槽释放依赖：后续batch输入Cast完成至VEC→MTE2释放相隔1036/1020模型周期。源码行debug_line缺失、实板临界路径仍未定位；模拟时间不代替实板性能。下一步候选为仅低精度独立转换路径提前释放输入槽，尚未实现。J026基线/J027条件候选不变。完整621文件归档取回、SHA/CRC/逐文件哈希通过；依赖J030自包含归档。详explorations/EXP_J031_SIMULATOR_910C_20261105.md。


# J032: release converted low-precision inputs earlier

From J026, only split-C FP16/BF16: move grads/features FreeTensor after both input Casts and the existing PIPE_V barrier, before weight Cast and math. No FP32 change because it aliases the original inputs. No queue depth, tile, UB, core count, arithmetic or output changes. J031 observed an input-lifetime gap of 1036 model cycles after the last input Cast; this motivates an actual-device test, not a performance prediction.
Compile; run 723 existing and new held-out boundary precision cases; verify every invocation in a bounded repeated stress runner; four alternating ABBA/BAAB rounds versus J026 using existing performance protocol. Use simulation only to check the event moved. Stop after this candidate. No installs, network changes, submission or push.


## J032：低精度输入提前释放，停止晋级

从J026仅移动低精度输入队列FreeTensor至输入Cast/原屏障之后，FP32机器码15入口全部不变。741精度、1344次逐调用输出压力、288组72000条实板profile全部通过。18组合四轮交替A/B均无四轮一致收益；[127,8192,1] FP16 12.887→12.797μs、BF16 13.062→13.084μs，变化未越过轮次/同码对照噪声。仿真证实输入最后读取至释放从1036降至3模型周期，但不代表设备收益。停止该方向，保留J026/J027状态和J032证据；完整归档已取回校验。详explorations/EXP_J032_EARLY_RELEASE_910C_20261105.md。


# J033: bounded hardware segment-metrics discrimination

No justified new production optimization follows J028/J029/J032. Test one previously unused tool path: installed CANN9 MetricsProfStart/Stop on arch2201, with msprof op KernelScale. J033E measures an empty window immediately before the second batch Compute; J033C brackets that Compute. Both retain the full J026 operator and all existing barriers. One pair per core, avoiding ambiguous repeated start/stop accumulation. FP16 and FP32 [127,8192,1], E/C/C/E order, warm-up100, selected call101. Verify outputs. Treat this as a diagnostic experiment, never a production candidate or uninstrumented exclusive time; empty-window and whole Task distortion must be reported. Stop if the tool fails or data does not separate the hypotheses. No new software, settings, submission or push.


# J034: one bounded low-precision split-C UB-layout experiment

J033 hardware KernelScale measured identical second-batch FP16 Compute windows in both rounds: 1548 cycles, bankgroup conflict ratio .211240, bank conflict .059432; empty-window6cycles. Official A2/A3 bank-conflict guide (CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0025.html) recommends 256-byte source-address separation. This is a new address-layout experiment, not J028 prefetch, J029 queue pairing, or J032 early free.

Baseline J026. Only low-precision split-C temporary buffer allocation changes. Round each float scratch extent to 512 bytes; add256B after mask, fg, gradsFloat, featuresFloat, dxFloat. This gives the simultaneous binary source pairs (grads/neg, features/grads, features/zero, zero/fg) opposite 256B phases modulo512, for every tile multiple64. No tensor offset/shape special case, no change to math/barriers/queue depth/tile/core routing. FP32 constexpr path unchanged. ExtraUB <=2816B; maximum low tile4288 uses <=152896B, below reserved available188416B. Physical bank read/write collisions are not claimed eliminated.

J034C adds the exact J033C second-batch segment instrumentation to that allocation candidate. Build J034 and J034C; three-dtype741precision and1344every-invocation stress; two reversed segment rounds versus J033C. If segment conflict/time does not improve, stop this candidate. If it does, four alternating ABBA/BAAB rounds against uninstrumented J026 on the existing18dtype/shape combinations. No parameter scan, no competition submission/push. Preserve every result.


## J033/J034 完成：实板分段定位与 UB 布局候选

J033 证实当前 2201 的 KernelScale 可用：空窗口 6 周期；FP16 第二批 Compute 为 1548 周期，bank group 冲突比例 21.1240%；FP32 为 1222 周期、26.0229%。

J034 仅调整低精度 split-C 临时 UB 地址布局，使二元源对相差 256B；数学、同步、tile、核数和路由不变。J034C 分段 FP16 从 0.938182 降至 0.766061 μs，BF16 从 0.984242 降至 0.812121 μs，bank group 比例降至约 0.15%，两轮复现。FP32 控制不变。

741+3 组精度及 1344 次逐次核对压力通过。普通四轮 ABBA/BAAB 共 21 组合、84000 条 profile：12 个修改组合中 6 个四轮均加速，无四轮均回退。目标 [127,8192,1] 的 FP16 Task 为 13.20475→11.75719 μs，BF16 为 13.26488→12.02356 μs。其余 6 个组合平均改善但有单轮翻转，包括新留出 [101,8256,1]；不宣称全面推广。

保留 J034 候选和 J026 对照，停止本轮单一实验；FP32 布局是有依据的下一步提案，尚未执行。完整报告：people/JadeYara/PReluGradUpdate_2329/experiments/J034/RESULTS.md；完整归档：EXP_J033_J034_UB_LAYOUT_910C_20261105。未提交竞赛、push 或修改网络设置。


# J035: independent FP32 split-C queue-address layout

Follow up J034, retaining all existing versions. J033/J034 hardware second-batch FP32 Compute was1222cycles/0.740606us, bankgroup conflict26.0229%, unchanged by the low-precision candidate. FP32 aliases queue tensors directly, so modify only its split-C allocation branch in a copy of J034. Round all five queue and two float temporary extents to512B. Add256B after grads and weights queues; round mask extent to512B and add256B. Source phases modulo512 become grads0, features256, weights256, dx/zero0, update0, neg256, fg256. All binary source pairs (grads/weights, features/grads, features/zero, zero/fg, grads/neg) have opposite256B phases. No arithmetic, barriers, queue depth, tiling, core count or dispatch change. Low-precision J034 path stays byte-identical by design and will be verified after compilation. This is one fixed architecture-based allocation rule, not a shape list or padding scan.

FP32 max tile5184: totalUB153344B (3008B over J034 allocation), below188416B available after8KiB reservation. J035C applies same allocation to J034C diagnostic instrumentation. First750precision cases (744existing plus6new),1728 per-invocation stress calls (27x64), then two reversed segment rounds versusJ034C across3dtypes. Only if FP32 segment conflict/time improves, run four alternating ABBA/BAAB complete Task comparisons versusJ034 on27dtype/shape combinations, including two fresh FP32 max-tile/tail shapes. Stop after this bounded experiment; no changes to J026/J034, system settings, competition or remote repository.


## J035 完成：独立 FP32 split-C 队列布局

基于J034，仅改FP32 split-C分配；FP16/BF16各15个函数字节完全相同，FP32只有split-C函数变化（6492→6608B）。750组精度、1728次逐次检查压力全部通过。最大FP32 tile5184时UB153344B，tile/核数/路由/数学/同步不变。

两轮分段：FP32 Compute 0.740606→0.560000 μs，bank group冲突26.0229%→0.2165%，bank冲突0→1.8398%；低精度控制不变。普通四轮27组合、108000条profile全部输出通过。8个修改的FP32组合中6个四轮均胜，2个平均改善但单轮翻转；目标127×8192从12.108188→10.794813 μs（−10.85%）。新留出107×15424四轮均改善；91×10368有一轮+0.159 μs。未修改FP16留出101×8256在完整混合顺序中平均+0.503 μs，已记录，不宣称全工作流加速。

保留J035候选及J034/J026，本轮仅一个优化，不再扫描。完整报告：people/JadeYara/PReluGradUpdate_2329/experiments/J035/RESULTS.md；原始证据：EXP_J035_FP32_UB_LAYOUT_910C_20261105。无竞赛提交、push或系统配置变更。


## J036 评估登记：冻结 J035，停止无依据扫描

范围仅个人 experiments/J036 交接文档、源码冻结包、manifest及共享台账/评估报告。不创建新kernel版本，不启动NPU作业。依据J035分段/完整profile与已排除路径评估，没有新的可靠关键路径假设；整理候选适用范围、TBE/平台口径及争议点复核待办。


## J036 完成：候选冻结与提交前待办

已核对J026/J034/J035全部源码文件与各自已测ZIP一致，两个版本差分均仅split-C头文件。J035已整合低精度与FP32布局，无需再次合并。个人experiments/J036保存J035-code.zip（SHA256 212183985a1795c76ca2fddf298d53ca4229321cdcdaa87517882b8ae17bbd35）、manifest.json与RESULTS.md。剩余Scalar/MTE2/Vector活动重叠，不据占比宣称唯一瓶颈；本轮无新可靠改动，未扫描或重跑设备。

提交前重点：确认910B模板与实测910C的目标口径、直接TBE对照、FP32两翻转点和FP16顺序偏移复核、按正式范围补精度、从冻结包目标环境构建及提交ID/SHA绑定。750精度、1728压力和108000条profile均为J035历史实测，本轮没有伪记新增通过。未提交竞赛。


# J037: submission-preparation validation, no kernel change

Read current CANNJudge task and own submission1109131. Inspect installed CANN9 TBE PReluGradUpdate and its ascend910_93 legacy registration. Build only a private test runner using aclopCompileAndExecute, with ASCEND_CUSTOM_OPP_PATH absent, for direct builtin TBE precision/performance against frozen J035. First a small FP32 pilot and six benchmark cases; if supported, four matched ABBA/BAAB rounds. Separately retest fp32[91,10368,1],fp32[147,6656,1],fp16[101,8256,1] as single-shape processes in four ABBA/BAAB cycles against J034, using exact existing inputs and250calls/window101-250. Preserve J035 source SHA212183985a1795c76ca2fddf298d53ca4229321cdcdaa87517882b8ae17bbd35. No submission, installations, network/security changes or source optimization.


## J037 完成：提交前同机 TBE 对照与争议点复测

J035 源码冻结。直接系统 TBE 小试及六点输出通过；四轮对照六点均快于 TBE。三个争议点独立进程四轮复核，结果及限制见 explorations/EXP_J037_TBE_READINESS_910C_20261105.md。96+48 组、36,000 条设备记录全部通过末次输出检查。旧 WA1109131 下载源码证实不同于保存的 P103/J001，不归因到 J001。完整原始日志和依赖校验已归档到 EXP_J037_TBE_READINESS_910C_20261105；没有提交或 push。


# J038: persistent zero vector in split-C

Follow J037, preserve J035. Quantified evidence: J035 target FP32 second-batch Compute924cycles; full target Scalar4.56us, Vector2.31us, MTE2 4.06us overlap. Existing split-C fills the same zero vector once per batch (6-7 per core for N127/groups20), despite fixed currentNum. J038 allocates one persistent zero tensor, initializes once in Init, retains every existing Compute barrier and all arithmetic/count-Select/Compare forms. No RepeatScalar, mask multiplication, queue/sync removal, tiling or core changes. Append buffer with256B leading offset so zero retains original dx zero source phase modulo512, preserving J034/J035 source-bank layout. Existing allocations unchanged. Additional bytes AlignUp(4*alignedCount,512)+256; max fp32 UB174592 and low170560, below188416. This reuses the upstream split-C invariant-lifetime idea already applied to weights, not a direct arch35 port. Previous cached-weight cast/prefetch/pair-queue/early-free attempts are not repeated.

Build J038 and instrumented J038C (same second-batch Compute markers as J035C); 750 existing regression plus6 new boundary/heldout cases, each-call pressure on33cases x64. Two reversed segment rounds across3dtypes versus J035C; full same-protocol4round ABBA/BAAB comparison versusJ035 on existing27 +6 fresh dtype/shape combos. Stop after this one fixed candidate, even if average noise looks favorable. All raw logs archived; no push/submission/network/system install.

Official references: https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0003.html (Duplicate/API); https://www.hiascend.com/en-sg/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0066.html (Compare count alignment). Existing upstream hicann/ops-nn b5f577d6 split-C keeps same channel weights across batch; J024/J026 already implement that. This new constant lifetime change is authored locally.


## J038 完成：split-C 常驻零向量

基于 J035，只把每 batch Duplicate 零向量移到 Init，附加独立零缓冲且保持原零地址模512相位；保留全部既有同步、tile/分核/路由/算术。仅 split-C 头文件改变；J035保留。756泛化精度、2112逐次压力通过；12组第二Compute分段显示FP32 924→857/858、FP16 1264→1163、BF16 1340→1227周期，两轮反向一致，但bank冲突比例上升。

33组输入、四轮ABBA/BAAB，共528组132000条ordinary profile，全部末次输出通过。28个实际改动组合平均都更快，17个四轮全胜、0个四轮全败。新留出[113,12864,1]三dtype四轮均胜；目标[127,8192,1]仅均值改善0.074～0.289us，均有反向轮次，不宣称稳定达标。未改路径亦有+0.036～+0.120us平均漂移。结果、全部表格、执行脚本、原始日志、归档校验在 explorations/EXP_J038_ZERO_REUSE_910C_20261105.md 及同名 logs 目录；个人源码/报告在 experiments/J038，插桩J038C仅诊断。不提交比赛、不push、不改网络/系统。下一项可检验假设是零别名消除后两次Select之间的同步依赖，本轮未改该同步。


# J039: remove one duplicate external Select barrier

Preserve J035/J038. Installed CANN9 interface kernel_operator_vec_cmpsel_intf_impl.h lines561-576 calls dav_c220 VselImpl(count). Actual mode2 implementation lines178-194 uses a temporary UB mask-address value, Duplicate/barrier, set_cmpmask/barrier, vsel/barrier, then ResetMask/SetMaskNorm. SHA9a0e66e77d4e6dac930b0a9913ba560695d8d29b9d08fdeaad41a4ca0a323705. Therefore the external PIPE_V immediately after first count Select is a duplicate candidate. J038 tensor reads/writes are disjoint (update<-mask,zero,fg; dx<-mask,grads,neg); but stronger evidence is Select's own end barrier, which also protects internal mask state. Correct earlier suggestion that zero-alias removal alone makes this safe; even J035's alias is already ordered by this API's internal barrier. Scope limited to actual CANN9/dav-c220 implementation, not all Select overloads/architectures.

Official PipeBarrier reference https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0271.html describes pipeline synchronization/automatic compiler sync. The live page body was not accessible through web extraction; installed official headers and device compilation are decisive evidence. No system headers modified.

Only remove external barrier between the two count Select calls in split-C. Keep pre-Compare, post-Compare, post-second-Select, casts, queues, all internal API barriers, zero lifetime, UB layout, tiling/routing. Build J039/J039C, compare ELF against J038,756 previous precision plus24 mask-directed cases across three dtypes/two shapes/four patterns (780total). Stress targeted24x128 plus existing33x64=5184 calls, outputs checked every invocation. Segment second Compute: two reversed rounds x3dtype. Ordinary J038/J039 same33inputs4ABBA/BAAB rounds250calls,window101-250. No competition submission/push/network changes. If identical machine functions, do not claim speedup; record compiler evidence and bounded control timings.


### J039 — 外层 Select 同步验证完成（2026-11-05）

- 只删除 J038 split-C 两次 count Select 之间的一条外层 `PipeBarrier<PIPE_V>()`；J035/J038 原版保留，未提交比赛、未 push。
- 本机 CANN9 官方 count Select 的 mode2 实现内部已有 vsel 后 PIPE_V；此证据比仅看应用层读写集合更直接。官方头文件、哈希和二进制证据已归档。
- 三 dtype 780 精度用例、5184 次逐调用双输出压力检查、12 次分段运行全部通过；528 组/132000 条普通设备 profile 全部完成并通过组末输出检查。
- FP16/BF16 全部15个 kernel 函数字节和地址不变，分段1163/1227周期不变。FP32 仅 split-C 字节变化但长度仍6748，分段857–858→866周期略慢。
- 33组合四轮全胜0、全负0；目标 `[127,8192,1]` J038→J039 FP32 10.886250→10.812313、FP16 11.833563→11.772375、BF16 11.902500→11.854688 μs，均有反向轮次。A/A −1.559…+2.360 μs，因此不认定微小均值差为加速。
- J039 作为正确性通过但无稳定收益的实验版本归档；保留 J038 候选与 J035 基线。下一项有依据的候选是公开 repeat Select 减少 count 接口内部重复 mask 状态配置，尚未实现或测量，不承诺收益。
- 报告：`explorations/EXP_J039_SELECT_SYNC_910C_20261105.md`；个人报告：`people/JadeYara/PReluGradUpdate_2329/experiments/J039/RESULTS.md`。完整归档、输入引用及校验记录见对应 explorations/logs 目录。


# J040: public repeat Select in split-C

Single experiment: preserve J035/J038/J039, derive J040/J040C from J038/J038C and change only the two count Select calls to public repeat overload. Keep both external PIPE_V after Select, Compare(+zero), casts, queues, zero lifetime, UB layout, tiling and routing. User explicitly authorized implementation and device testing, no competition submission or push.

Installed CANN9 dav_c220 Select header SHA9a0e66e77d4e6dac930b0a9913ba560695d8d29b9d08fdeaad41a4ca0a323705: count mode2 lines159-198 uses SetMaskCount, mask setup, vsel, final barrier, ResetMask/SetMaskNorm. Public repeat lines252-264 calls mode2 Pre (temp UB mask address, Duplicate/barrier, set_cmpmask/barrier), SetMask<float>(64), vsel(repeats,strides,mode2), without count-mode switch/restore or ending barrier. Retaining caller barriers protects both temporary mask-register reconfiguration and output dependencies. Do not use private intrinsics or isSetMask=false.

cmpCount already AlignUp(currentNum,64), buffers allocated to alignedCount. Float computation applies to all three storage dtypes. Use mask64, repeats=cmpCount/64, strides1/1/1 and8/8/8 (32B blocks -> contiguous256B float repeats). Max split-C cmpCount5184 =>81repeats, low4288=>67repeats. Mode2 receives same mask base afresh for each Select and traverses mask with repeat data; targeted alternating64/lane/batch/channel-tail patterns test this hardware behavior. Existing padded tail handling and CopyOut(currentNum) remain. This is not mode0 which would reuse one SPR mask across repeats.

Reuse J039's exact780 finite-data regression inputs including24 mask-targeted inputs;57 stress cases5184invocations with both outputs checked every call. Build J040/J040C, inspect ELFs, two reversed second-Compute segment rounds across3dtype againstJ038C. If correctness passes, same33cases fourABBA/BAAB ordinary device timing rounds250calls/window101-250 againstJ038. Stop this fixed candidate after matched results; no indefinite scans. Archive raw evidence with SHA-bound historical input dependencies.


### J040 — 公开repeat Select验证完成（2026-11-05）

从J038派生，保留两次Select后的PIPE_V，仅改公开repeat接口：float mask64、cmpCount/64次、stride1/1/1/8/8/8。实际CANN9/dav-c220接口和stride/Compare实现快照已保存，未修改系统头文件。J035/J038/J039保留，未提交比赛或push。

780精度用例、5184次逐调用双输出压力检查全部通过。12次反向分段结果：FP16 1163→1162周期，BF16 1227→1229，FP32 857–858→873。三dtype仅split-C函数字节变化，其余14函数字节相同。

33组合四轮ABBA/BAAB，528组/132000条profile及组末输出全部完成。28个实际修改路径19个四轮均快；但5个未修改控制组合也全部平均更快、2个四轮均快。目标[127,8192,1]均值FP32 10.939875→10.611125、FP16 11.804750→11.572313、BF16 12.039875→11.684875 μs；C=1控制也下降约0.19–0.28 μs，因此不能将整体差值归因于repeat改法。J040暂不替换J038，保留实验版本与混杂证据。停止该固定候选的参数扫描。

报告 `explorations/EXP_J040_REPEAT_SELECT_910C_20261105.md`，个人目录 `people/JadeYara/PReluGradUpdate_2329/experiments/J040/`。用户追问的小通道数据已从J035/J038/J039已有摘要与输入哈希核对提取，见个人目录 `SMALL_CHANNEL_HISTORY.md`，未新增测试；近期只有N600000/C1/S1三dtype的性能样本。


# J041: small-volume, small-channel latency measurement

User requested representative small N/C/S data across FP32/FP16/BF16. No kernel change, no rebuild, preserve J035/J038/J040 decisions. Select16 existing legal shape/weight configurations x3dtypes from archived780-case regression: total6..9001 elements, C1..33, N1..127, S1..9001. Include63/64/65 scalar-channel boundary, C31/32/33 S1 boundary, multi-N/C/S shapes, and shared/per-channel weights at[4,6,8]. N1C1S65 uses existing mixed seed grid input instead of all-positive/negative/zero-only samples. Hash-bound exact48 inputs and golden, no generation.

Use already built ascend910_93 J035/J038 packages and runners, verify source and package provenance against previous J038-layout-campaign. Run explicit48-case precision for each version (96checks), then fourABBA/BAAB rounds with250calls percase/window101-250; lasttwo rounds reverse case order, eachround same48 inputs. 48x16=768groups and192000ordinary msprof profiles. Everygroup finaldx/update checked; this is not a new every-invocation stress suite. Record Task Duration and aiv_time separately, perround range, sameversion A/A-B/B spread, actual core count. No large-input mixture in this campaign. Stop after fixed suite, no score submissions/push/network changes.

Report actual warm/reuse device-kernel latency, not hostend-to-end or leaderboard. These small paths do not use split-C modified by J035/J038, so do not invent small-path speedup from noise. Archive self-contained selected inputs, source/packages/runners and all raw profiling/logs. J040 remains a correctness-passing experimental repeatSelect version with confounded performance, not promoted.


# J041 PAUSED BEFORE DEVICE LAUNCH

User superseded the small-input test request: pause new algorithms and tests, review existing large-channel evidence first. Only local48-case selection, scripts, and a read-only remote status check completed. launch_j041.py was never executed; no J041 remote job, precision run or performance result exists. No kernel source changed. Last observed NPU status2026-11-05T08:52:35+08:00: no running processes; J040-layout/perf exit0. Do not run the prepared scripts until the user resumes. Existing J040 conclusion unchanged. Current work is documentation-only review; no small-channel/big-channel tradeoff implemented.



## 2026-11-05 J041 暂停后：大通道已有证据审阅完成

用户要求先审阅取舍，暂停新增算法和补测。J041仅本地48组选点/脚本准备，未启动设备任务；J040作业已退出0，最后只读检查08:52:35 +08:00 NPU无运行进程。本轮仅整理文档，不改源码、路由或候选状态。

审阅：[REVIEW_LARGE_CHANNEL_910C_20261105.md](explorations/REVIEW_LARGE_CHANNEL_910C_20261105.md)。保留个人J041副本及318条同批对照（30条[127,8192,1]三dtype目标对照）、10个分析源文件SHA256。逐版策略、失败边界、旧小S=1顺序回退和近期小总量计时缺口均已列明。J035冻结基线/J038增量候选保留；J039/J040未晋升。没有预先实施以小通道换大通道的取舍，无比赛提交、推送或新设备测试。


# J042：split-C 的 TPipe 移到类外

用户恢复优化并撤回官方提交。唯一源码变量：以J038为基线，split-C类中的TPipe对象改为指针，在入口的编译期split-C分支内先创建TPipe，再创建算子并传入Init。其余分支保持原入口，数学、同步、UB分配顺序/大小、队列深度、tiling、分核、权重常驻、CacheMode、Compare/Select全部不改。保留J035/J038，不推送、不比赛提交。

依据：同批J038目标[127,8192,1]的Scalar活动为FP32 4.5555、FP16 4.8981、BF16 4.8392us；不是独占关键路径。Ascend官方API优化参考说明类内TPipe初始化暴露类地址可能阻止Scalar常量传播，类外TPipe是可验证的编译优化假设，不把其他算子的17%例子套到本题。
https://github.com/Ascend/agent-skills/blob/master/skills/ascendc-operator-performance-optim/references/api-usage-prof.md

先真实910C构建和780组既有三dtype精度、57组5184次逐次输出压力。查ELF是否只改变split-C。然后J035/J038/J042三版四轮成对位置顺序：ABC CBA、CBA ABC各两次；每组250调用、取101–250。30组输入：4个小shape×3dtype + 6个大shape×3dtype，混合排序、后两轮反序，保存每组输出与profile。小输入[1,1,65]w1、[2,3,1]w3、[4,6,8]w6、[127,32,1]w32；大输入[127,8192,1]、[113,12864,1]、[91,10368,1]、[147,6656,1]、[107,15424,1]及[600000,1,1]，均复用已有输入hash。此为一项有界实验；无稳健收益则保留基线并结案，不继续扫参数。

## 用户建议逐项对照

| 建议 | 本机实现/证据 | 本轮处理 |
|---|---|---|
| 分核、尾块 | J011小标量分核；J025失败选择规则；J026限定split-C范围；batch用整数区间均分、尾通道min截断 | 不重复扫核数；沿用精度中的尾块/边界 |
| L2 | 默认NORMAL；J030 kernel/application replay命中差异7%/约100%，不能混作一种计时；SetL2CacheHint本机存在 | 不全局关L2；没有直接证据表明本输入关闭会提速 |
| 权重常驻 | J018整C；J024/J026按C块跨batch已实现；低精度转FP32缓存J027未稳定改善 | 不重复已做方案 |
| DataCopyPad / NDDMA | 当前Pad支持尾块；P102对齐DataCopy切换退化已撤；本机CANN9 MultiCopy/NDDMA声明和实现仅__NPU_ARCH__3510/5102 | 不把NDDMA用于dav-c220；不重做P102 |
| TQueBind | 本机接口存在；参考主要消除纯搬运的UB→UB拷贝，当前输出由Mul/Select生成，没有该冗余直拷 | 不机械替换；若复用输入输出需独立别名/生命周期实验 |
| UB bank | J034低精度、J035 FP32地址布局已见实板Compute收益 | 保持布局，避免改动混杂 |
| counter | 当前count重载已内部配置count模式；J040 public repeat Select未获得可归因稳定收益 | 不把已启用功能当新优化，不重做J040 |
| gather | 当前split-C是连续GM块读取、同通道权重驻留，不存在逐元素非连续索引读取 | 暂无替换机会，增加gather无依据 |
| 队列深度 | 当前1；J007/J028预取双缓冲无稳定收益或诊断回退 | 本轮不动深度/UB预算 |
| FreeTensor | J032提前释放低精度输入：模拟等待缩短未转化实板收益；FP32存在活跃别名 | 不重复提前释放，不删同步 |

J038 MTE2/Vector活动比：FP32约1.95，FP16约0.93，BF16约0.94。不同流水可重叠、各列最大核未必同一核，不能相加当总延迟或占比。910C实板核内Instr/PipeTimeline不可用；模拟器只定位依赖，不提供实板时间。API核验日志在本轮归档。


## 2026-11-05 J042 结案：split-C 类外TPipe

用户取消官方提交并恢复优化。本轮只把J038的split-C TPipe移到入口局部变量，保留数学、同步、分核、tile、UB布局、队列深度；J035/J038未改。910C/CANN9三dtype编译成功；780精度、5184次逐次压力通过；30组小/大输入三版本四轮交替，720组/180000条设备profile，全部输出检查通过。各dtype仅split-C函数变化，其余14函数字节一致。

目标[127,8192,1] J038→J042均值FP32 10.7321→10.4523、FP16 11.5510→11.4300、BF16 11.7394→11.5545μs，但四轮方向翻转，未改控制也同向漂移；FP32 Scalar四轮下降，整算子收益仍未稳定归因。J042不晋升，保留实验。用户建议逐项对照、真实MTE2/Vector活动比及910C timeline限制见EXP_J042；NDDMA本机接口仅3510/5102，不适用dav-c220。未比赛提交、未push，无性能取舍实施。


# J043：split-C 轮转 batch 分配

会话：JadeYara，PReluGradUpdate_2329；唯一目标是验证 split-C batch 分配拓扑改变能否缩短大输入延迟。允许个人 experiments/J043 与本题台账、报告、日志；用户明确授权继续实机优化，不推送、不提交。保留 J035/J038，J042 未晋升，不继承其 TPipe 改动。

## 依据与单变量

J042 原始 profiler 的 Block Num 确认 J038 [127,8192,1] 三 dtype 均 40 核。由 Host/kernel 推导：2 通道块 × 20 batch groups，tile4096，每核6/7 batch；不是少核运行。[91,10368,1] 低精度 39 核，3×13，每核7 batch；FP32 40核，2×20，每核4/5 batch。每核具体完成时刻不可由当前工具还原。

J025P/J026 已覆盖 26 N/C ×3 dtype，20/32/40 核和全/半 tile，三轮扫描；其机器码早于 J034/J035/J038，因此不能称当前版本核数已穷尽。目标 [127,8192,1] 历史 FP16 40核全tile12.587us、32核12.713、20核16.926、40核半tile14.507；FP32 40核11.667、32核11.620，差异很小。此前没有测 batch 连续/轮转映射。

J038 原分配：core 的 batchGroup=g，n=[floor(N*g/G), floor(N*(g+1)/G))。J043 改为 n=g,g+G,...<N。通道块分配、blockDim、tile、UB、数学、同步、权重常驻和拷贝总字节数均保留。只修改 splitc.h 的 batch 起止及循环步长。

假设：目标 N127/G20 的同通道块首批逻辑 batch 编号原为0,6,12,...120，新为0..19，可能改善同时活跃 GM 地址的局部性。首轮是逻辑迭代，不保证硬件各核锁步；该实验也会改变每核连续访问距离，可能退化。J042 同轮 J038 FP32 MTE2活动4.252us、Vector2.212us，支持优先检查搬运相关变化，但不能认定GM局部性或HBM bank冲突。两个活动计数可重叠，不能当作独占延迟比例。

## 有界验收

先复核现有780用例索引覆盖及各核负载分布，再 910C 编译、780精度、57输入5184次逐次输出压力。三 dtype。只测一个候选，不扫描额外核数或tile。

性能复用 J042 的30输入，J035/J038/J043，四轮 ABC-CBA/CBA-ABC 两轮循环；后两轮反向输入顺序。每组250次，取第101–250次任务时间中位数，每轮同版本两位置平均，再比较四轮差值和未改路径漂移；记录全部原始profile及Scalar/Vector/MTE2/MTE3。12小输入为[1,1,65]w1、[2,3,1]w3、[4,6,8]w6、[127,32,1]w32各3dtype；18大输入为[127,8192,1]、[113,12864,1]、[91,10368,1]、[147,6656,1]、[107,15424,1]、[600000,1,1]各3dtype。目标区有整除、不整除batch及通道尾块。不是新增留出集，不声称官方评分。预期总操作20–30分钟；精度失败先停止性能，若收益不稳定则保留为实验且结束本方向。


### J043 结案：split-C 轮转 batch 分配，未晋升

确认J038目标[127,8192,1]三dtype实测40核（2通道tile×20组，每核6/7batch）；J025已扫20/32/40核及半tile，本轮未重复扫描，只改连续batch区间为轮转分配。J038来源，未继承J042。780精度、57输入5184次逐次双输出检查PASS；30输入四轮三版本交替720组/180000条设备profile，Block Num均一致。

目标J038→J043均值：FP32 10.6210→10.2846、FP16 11.3798→11.2174、BF16 11.6148→11.3670μs，但第四轮都回退；13个改动输入0四轮全胜。未改控制路径有相同顺序漂移，MTE2首轮下降未稳定复现。不晋升，不替换J035/J038，不追加扫描。小输入未发现四轮一致回退，精度全通过。

报告：people/JadeYara/PReluGradUpdate_2329/experiments/J043/RESULTS.md；共享EXP_J043_CYCLIC_BATCH_910C_20261105.md；完整构建/精度/profile与SHA依赖归档位于explorations/logs/EXP_J043_CYCLIC_BATCH_910C_20261105。官方输入与评分未获得；未提交比赛、未推送。


# J044：计时方法校准，无算子改动

JadeYara / PReluGradUpdate_2329。本轮唯一目标：定位J043顺序偏差的可观测来源，验证无profiler与必要profiler对照。允许个人J044测试入口、本题台账/报告/日志；不改J035/J038/J043源码/包，不碰提交页面、不提交、不push，不更改网络、安全、频率或系统配置。

原J021 runner逐次GetWorkspaceSize、launch、SynchronizeStream；J043每版独立进程，100次预热并未消除相同机器码控制的位置偏差。不能据Task减aiv_time命名根因。

本机CANN9 ACL1.17已声明EventElapsedTime、TIME_LINE和RICapture接口。官方9.0文档支持A3：
https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_0090.html
https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_0080.html
https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1782.html
事件计时输出毫秒，TIME_LINE自身有开销；捕获回放减少主机投递开销，但改变为连续执行条件，不能冒充官方单次kernel延迟。

阶段1：J038单输入FP32大目标pilot，比较sync、enqueue、capture-graph，32/128批长度，记录CPU wall和device事件区间；核对每阶段输出和捕获是否可用。只改新runner，按需编译，不安装。

阶段2：方法有效后先做J038同版本A/A，三dtype，小[2,3,1]、C32[127,32,1]、大[127,8192,1]、未改控制[600000,1,1]；变化顺序、预热、批长度。无profiler连续回放的事件span/次数作为热态吞吐口径，长度差分斜率用于检查固定边界开销。同一机器码A/A四轮大输入差应在0.10us或2%内才用于微小候选比较；失败则报告精度下限，不继续算法扫描。

阶段3：必要的少量匹配profiler/no-profiler对照，原样sync与graph，严格相同输入、预热和计时入口；分析Task/aiv_time/活动计数，保留全部原始数据。空闲频率只作空闲读数，不能推断单个kernel实际频率。若方法可辨识，再以J035/J038/J043未改包做有界对照；否则只给方法结论。

每次运行前检查旧作业；每阶段核对输出。优先原有输入/hash，不生成新算子代码。原始日志/CSV/JSON与来源指纹归档。


J044追加单变量控制：同输入同进程同地址仍出现约0.5us阶跃，1Hz采样活动芯片1650MHz未解释该波动。现有graph计时每窗口重建模型，新增cachedgraph只捕获一次、跨窗口复用，三次独立进程复现同FP32大输入。只改变新测试入口的模型生命周期，不改算子或系统。该对照只区分新入口的模型重建因素，不能自动解释旧J043非捕获运行的全部漂移。


# J045：固定 tile/核数的连续通道 tile 遍历

JadeYara / PReluGradUpdate_2329。响应用户架构复盘与本人提交安排；本轮只做一个可证伪架构对照，不操作在线工程、不提交、不推送。

历史核验：J023的S=1路径已经向量DataCopy权重，不是P105逐元素展开；J024与它有同机对照且大目标明显改善。但两者同时改变tile切分/任务分布/UB布局，尚无当前J035固定tile/核数/Compute的严格flat对照。因此不能说flat从未测试，也不能直接把旧J023的17us作为此候选结论。

基于冻结J035，唯一文件op_kernel/p_relu_grad_update_splitc.h。保留Host分支条件、blockDim、tileLen、UB分配顺序/大小、CopyIn/Compute/CopyOut及同步数学；只改任务遍历与权重驻留策略：原每核固定C块跨batch，改为线性通道tile连续区间，每tile向量重读权重。使用既有nonresident队列释放逻辑，无逐元素权重展开、无Gather、无新设备API。

ctiles=ceil(C/tileLen)，totalTiles=N*ctiles，每核flat区间[floor(totalTiles*core/cores),floor(totalTiles*(core+1)/cores))；每tile在当前C块内，尾块截断；当前输出段后紧接下一段，跨batch时权重offset归零。保持tile数量和核数；有C尾块时每核字节负载分布可能改变，不将其混作仅缓存效应。

对[127,8192,1]仍40核、tile4096、每核6/7tile。权重读取由20*C增至127*C，逻辑GM总元素由4*N*C+20*C变为5*N*C，增加20.27%；如果连续遍历仍更快，才支持跨batch调度/地址组织的代价超过权重节省。若变慢，则保留split-C，不靠未知官方shape推翻本轮结果。不能从GM逻辑字节直接推算HBM带宽。

先索引覆盖检查及三dtype780精度、57输入5184次逐次双输出压力；通过后J035/J045同机交替，普通msprof Task及少量无profiler enqueue事件区间分开记录。J044已知跨进程约0.4–0.6us漂移；细小均值变化不宣称收益。保留小输入和不走split-C的大输入控制，不用有Block Num异常的图PMU。

官方1148578由用户报告Pass，5点7.15/7.29/7.11/5.88/6.06us；其实际全量源码未取得，不能绑定J035。若完整文件指纹等于J035，候选只需替换上述一份header；若不等，先取得1148578完整回退包，不能把单header混入未知Host/tiling ABI。交给用户的候选会含完整源包、单文件diff、SHA清单及条件化替换/回退步骤，平台由用户操作。


J044 已完成：五阶段均退出 0，发现 graph PMU Block Num/归一化异常及跨进程偏移；cached graph 未运行。见 explorations/EXP_J044_TIMING_REVIEW_910C_20261105.md。未改算子/未提交。


J045 已完成：固定 J035 Host/tile/核数/UB 的 flat 通道块遍历与逐 tile 向量重读权重对照。780 精度、5184 压力、480 profiler 组/120000 kernel、192 无 profiler 窗口通过；多个大通道稳定回退，不升级基线。J044 确认计时漂移。用户确认官方 1148578=J035，工具未下载提交快照；精确单头文件/回退/hash 包在 J045/manual-platform-trial，用户自行决定是否做诊断提交。纯权重生命周期对照尚未执行。无在线改动/提交/push。见 explorations/EXP_J045_FLAT_CHANNEL_TILES_910C_20261105.md。


# J046：同切分/同遍历的权重生命周期对照

基于冻结 J035，仅将 split-C residentWeights_=true 改 false，并删除 Process 末尾驻留 tensor 的释放。保留 Host、Ctile、batchStart/End、batch 遍历、blockDim、UB 分配布局、算术、低精度每 tile Cast、其他路径。使用原 nonresident 的每 tile 权重 DataCopy/EnQue/DeQue/FreeTensor。改变的是权重搬运和队列生命周期整体，不能把效果仅归因于 HBM 字节。

先 780 三 dtype 精度 + 57 案例 5184 次双输出压力，再沿用 J045 的 30 个小/大输入并加入两种 C 尾块形状三 dtype，共 36 个输入，四轮 ABBA/BAAB，后两轮反转输入顺序。普通 PipeUtilization 记录 Task、AIV/Scalar/Vector/MTE 活跃时间、cycle、Block Num；未改路径作漂移控制。有限无 profiler enqueue 对照用于交叉核查，口径独立。统计权重搬运/队列次数为基于执行路径的逻辑计数，不冒充 PMU 实际 HBM 事务；不预设 L2。

用户已确认官方 1148578=J035，官方输入未知，用户负责提交；本轮不操作平台、不提交、不推送。所有原版本和证据保留。


J046 完成严格权重生命周期对照：J035 相同切分/遍历/核数/UB/计算，仅权重逐 tile 重读释放。780 精度、5184 次压力通过；36 输入4轮普通 profiler 的19个 split-C 输入全部四轮回退，无 profiler 的6个适用输入也全部四轮回退，保留 J035、不采用 J046。附16份实板 MemoryDetail，逻辑搬运计数与硬件计数分列。首次 counter 因 msprof 私有目录检查未执行采集，保留日志后在新建私有目录成功恢复，不改现有权限。未操作平台/未提交/未push。见 explorations/EXP_J046_WEIGHT_LIFECYCLE_910C_20261105.md。


# J047：复核 J027 转换缓存与 J035 UB 布局的交互

先审旧证据：J027 基于 J026，仅 split-C FP16/BF16 将权重首 tile Cast 到既有 negFloatBuf 常驻；Compare 后复用 featuresFloat 写 weight*grad，再 Select dx。708精度通过；普通63输入无稳定收益，固定A/B/A目标低精度约-0.85/-0.766us，但未改FP32也-0.326/-0.341us。旧kernel/application replay仅J026/J027IO，没有J027，不能据此声称缓存候选已经做过冷/热对照。

实质遗漏条件：逐字比对46文件，J026→J035唯一差异是splitc.h的UB分配布局。J034实板Compute bank-group冲突和时间下降，改变了当前基线；旧J027从未与新布局组合、也未做对称ABBA与无profiler对照。一次有界2x2：旧布局J026/J027，新布局J035/J047；候选只把已经验证的J027 Compute改动原样移植到J035。保持Host/tiling/核数/队列分配/输入遍历/权重GM驻留，FP32不改。新布局保持Mul/Select双输入0/256相位，实际收益待实测，不能把相位推导当硬件计数器。

780三dtype精度和57案例5184逐调用压力；性能仅15输入=三个大形状[127,8192,1]/[81,8256,1]/[91,10368,1]与两个小控制[127,32,1]/[4,6,8]×3dtype。每布局内部四轮ABBA/BAAB，交替布局顺序、后两轮反转case顺序。用9输入无profiler enqueue复核，保持协议分开。只实施这一候选，不继续扫描；若无可复现收益则停止权重转换缓存方向。没有新的MemoryDetail或仿真作业。用户负责官方提交，本轮不操作平台/不push。


J047完成J027复盘及布局×转换缓存2x2：旧replay仅J026/J027IO、固定ABA存在字节未改FP32假收益；确认J026→J035唯一变化为splitc.h UB分配。J047原样移植J027 Compute到J035。780精度、5184压力、15输入480组120000普通profile、9输入288无profiler窗口通过；旧布局低精度连续提交有四轮一致局部收益，但新布局变小/尾块协议翻转，不升级J035。停止该复用缓冲方案追加扫描；下一独立FP32权重buffer/保持原调度仅设计未运行。未提交/push/在线改动。见explorations/EXP_J047_CAST_CACHE_LAYOUT_910C_20261105.md。


# J048：独立 FP32 权重缓存，保留 J035 乘积缓冲与计算顺序

唯一目标：隔离 J027/J047 中权重转换缓存与乘积缓冲复用/调度变化的耦合。以冻结 J035 为基线，只改低精度 split-C：在 UB 分配末尾增加一个 FP32 TBuf，首 tile Cast 权重到该缓冲；原 Mul 仍写 negLocal，原指令顺序与显式 PIPE_V 屏障均保留。FP32、Host/分核/tiling/队列/原缓冲地址不改。

低精度原 UB=11A+6F+1280，新增 F 后 11A+7F+1280，A=AlignUp(tileLen,64)，F=AlignUp(4A,512)。历史最大 A4288 时170304/188416 bytes；目标 A4096 时161024。进一步核实际 tiling/测试边界。新增缓冲的压力是本方案自身代价，不能宣称只有指令数量变化。

范围：780 三 dtype 泛化精度、57 输入5184逐调用压力；15个大小输入同机四轮ABBA/BAAB，250次取101–250中位数，9输入无profiler连续enqueue单独复核。检查FP32/小路径二进制不变、Vector/Scalar计数与轮次漂移。没有稳定收益就保留J035，不制作平台替换包。用户自行决定官方验证；不操作平台/提交/push。只写 JadeYara/J048 与本题实验记录。


J048完成：独立FP32权重TBuf保留J035乘积缓冲/计算顺序/屏障；最大低精度UB170304/188416。780精度、5184逐调用压力、15输入240组60000 profiler记录、9输入144无profiler窗口通过。低精度目标/尾块无profiler四组均四轮改善；普通profiling BF16尾块四轮回退，其他大输入有轮次翻转。保留J035；已备用户手动官方一次诊断包及唯一头文件/回退/hash，不自动推广。FP32 split-C机器码+4bytes，其余小路径不变。未操作平台/提交/push。见explorations/EXP_J048_DEDICATED_CAST_CACHE_910C_20261105.md。


J048官方反馈（用户提供，提交编号未知）Pass，8.42/8.35/8.41/5.75/5.85μs；对J035 1148578前三点+1.27/+1.06/+1.30μs，后两点-0.13/-0.21。撤销整体替代建议，核验J035回退头SHA f4c5f793a36a152c5ee3cd4208c359cefc3246eaacba8a930916e808bd2c1ead，用户自行恢复提交。本轮只读复核及文档更新，无新设备作业/平台操作/push。下一候选仅设计：编译隔离原FP32类，或新增UB不缓存控制；不猜官方shape。详explorations/EXP_J048_OFFICIAL_REVIEW_20261105.md。


# J049：编译隔离 J035 FP32 与 J048 低精度缓存

用户要求继续优化，提交由用户负责。本轮只消除J048新增成员对FP32编译产物的连带改变，不猜官方前三点dtype。J035为基线，原split-C头文件完整保留。新增独立命名空间的J048缓存类，kernel入口只让half/bf16 split-C使用新类；float继续使用完整原J035类，不增加运行时判断。

先验证全部FP32入口机器码/地址恢复J035，并记录低精度是否等同J048。780三dtype精度、57输入5184逐调用压力；15大小输入四轮ABBA/BAAB普通profiling、9输入无profiler连续enqueue对照J035。现有J048官方回退依旧有效，不能把本轮自选性能映射官方五点。若FP32机器码不能恢复，则不将候选作为排除其连带变化的对照。

本次低精度UB/运算/分核与J048不变，仍最大170304/188416字节；J035已验证FP32恢复其原分配。仅写JadeYara/J049和本题台账，无安装/网络变更/平台操作/提交/push。保留所有版本。


J049完成：编译隔离，原J035 FP32类与J048低精度缓存类分开。实证整个FP32 kernel.o=J035，FP16/BF16整个.o=J048，主机4库路径+3runner可执行段亦与J035一致。780三dtype精度、5184逐调用压力、15输入240组60000 profiling、9输入144无profiler窗口完成。低精度目标均值仍有收益但轮次/协议翻转；相同FP32机器码第三轮Task仍差+0.955/+1.080μs，AIV也增加，不能将其当纯计时器错误或把小均值差当确定优化。J035继续为基线；J049仅提供用户手动官方编译隔离对照，未证明官方前三点恢复、不猜shape/dtype。J048官方退化复盘与回退SHA已记录。未操作平台/提交/push；无更多设备扫描。见explorations/EXP_J049_FP32_ISOLATION_910C_20261105.md。


J049暂停：用户升级云端软件。4个云端作业均此前退出0，下载也自然完成；停止检查无仍运行的本任务下载/SSH进程。用户暂停后未再发起云端命令，仅本地保存结果。等待用户明确继续后再核升级环境。详explorations/EXP_J049_PAUSED_FOR_UPGRADE_20261105.md。


# J050：CANN9.3环境重建J035与A/A校准

用户明确同意恢复，并依次进行A/A校准、额外缓冲与转换缓存A/B/C、按每核复用次数评估条件分派。官方提交由用户负责，不操作平台/提交/push。

已只读核实：/home/developer/Ascend/cann解析为cann-9.3.0，安装版本9.3.0，compiler/OPP组件9.3.0-beta.1；驱动仍25.5.5；旧J049四阶段退出0，NPU无进程。记录新的compiler/msopprof版本，不将9.0数据混作9.3对照。

J050 code是冻结J035的逐字副本，源码包SHA212183985a1795c76ca2fddf298d53ca4229321cdcdaa87517882b8ae17bbd35。在全新J050构建目录针对ascend910_93编译，验证780有限值三dtype精度和5184逐调用压力。工具或API不兼容时仅修复本任务构建/测试适配，不安装组件、不更改系统库或旧结果。

基线通过后A/A：同一版本、同一进程、同一输入分配，ABBA/BAAB标签；记录输入/输出地址、调用事件与host时间。使用无profiler连续enqueue和必要普通profiling分别核查；不使用此前有BlockNum问题的graph PMU作为性能依据。先量化同进程/跨进程噪声，后续候选收益需与该误差范围区分。

A/B/C下一阶段定义：A原J035，B新增独立FP32权重缓冲但每tile转换，C相同缓冲布局只首tile转换；B/C均隔离原FP32类。A/B同时包含初始化/额外UB/权重读取地址变化，不冒充只有初始化一项；B/C隔离缓存策略及其分支。复用次数以实际N、Ctile、batchGroups导出，先探索和留出形状验证，不使用官方测试编号特化。


# J051/J052: CANN 9.3 A/B/C weight conversion isolation

A=J050 (exact J035 source rebuilt on 9.3). B=J051: independent FP32 weight buffer, conversion on every tile. C=J052: same buffer/layout, first-tile conversion cache (exact J049 source rebuilt on 9.3). FP32 retains the original J035 class. A/B includes UB allocation, initialization and operand address/in-place differences; B/C isolates conversion caching and associated branch scheduling. No official-shape inference or platform action.

J050 baseline passed 780 precision cases and 5184 checked invocations. Same-process A/A ordinary profiling maximum delta 0.0195 us; enqueue reached 0.467 us and is not used alone to accept gains. Direct binary launch was investigated but not executed: generated metadata opParaSize=88 differs from the visible seven-pointer signature; ABI is not yet verified. Use balanced ordinary profiling with repeated A controls, log all five allocation addresses, and treat cross-process allocation/state variation as a remaining confounder. Only repeatable differences exceeding controls justify a dispatch rule. Cover small controls and split-C N/C/dtype reuse ranges with holdouts.


## J050–J052 completed on CANN9.3.0 / beta.1 components
J035 rebuilt unchanged (J050); J051 separate FP32 weight buffer per-tile conversion; J052 first-tile cache. Three variants each pass780 finite precision +5184 checked stress calls. J050 same-process profiling A/A max0.0195us, enqueue max0.466992us. ABC27cases×6rounds×4windows,226800kernel records all precision pass. Cross-process duplicate A max1.041us; FP32 whole objects identical yet timing moves. Vector cache saving increases with reuse; high reuse FP16/BF16 C/B six-round gains~0.795/0.815us, C/A still flips one round. No dispatch adopted; J035 retained. Full report explorations/EXP_J050_J052_CANN93_ABC_20261105.md. No platform/push. Archive SHA ec5032a50c72932b59039a3216c814bf1715413e047f4dffd0efd9160e50e4ac, fully local; CRC, binary dependencies and source provenance verified.


# J053: same-process, fixed-allocation A/A and A/B/C

User authorized continuing the J050-J052 timing diagnosis. One goal: compare the CANN9.3 rebuilt J035 baseline with independent-buffer per-tile conversion and cached conversion while holding process, tensors and stream fixed. No platform operation, submission, push, network/settings changes or installation of new software.

Test-only aliases PReluGradUpdateA/B/C in one private package derive mechanically from J050/J051/J052. Only registration identifiers and file names change. Require compiled function-body SHA correspondence to original variants before interpreting timings; preserve originals. This avoids guessing the existing binary's hidden direct-launch ABI.

Correctness: all780 finite three-dtype inputs and5184 per-invocation stress checks for each alias. Same-process A/A labels invoke the same A function. Ordinary profiling keeps prior timing boundary: each kernel is synchronized;350calls/window, discard first200, median last150 Task Duration and AIV metrics. All five device addresses logged; no allocation/free between variants of a case. Predeclare balanced variant orders, randomized case order, low/high reuse, tail and small controls. Retain per-round deltas and A/A spread. Pipeline active times overlap and are not exclusive latency. A/A or unchanged controls outside0.1us/2% limits require explaining the drift before accepting a dispatch. If a generic reuse rule is supported, validate held-out shapes and original-path precision before preparing a user-operated replacement and rollback. No official-test-number specialization.

Writable scope: people/JadeYara/PReluGradUpdate_2329/experiments/J053 and this task's experiment records; cloud /home/developer/2329-JadeYara-20261104-1812/J053*.


## 2026-11-05 J053 接手：同进程测试入口（仅本地）

按用户“用 stop-that-shit，只做当前必要的优化”收窄范围：只补 J053/control 的 C++ A/A/B/C 入口、CMake 和 Python 计划/汇总脚本。J035 保留提交基线，Host/Kernel 未改。每个 case 复用五个张量和同一流执行六种平衡顺序；复用已有精度/压力/27-case 输入，压力逐次检查双输出；不调整 golden 或精度阈值。

35 个别名源文件与 J050/51/52 的机械改名映射检查通过；6 项 CPU 合成验证通过。最初写 test_campaign.py 失败（Access denied），对应 unittest 导入失败；之后以内存执行 CPU 检查，没有重试写被拒绝的文件。详见 explorations/logs/EXP_J053_LOCAL_20261105/local-validation.log 和个人 experiments/J053/control/README.md。

状态：本地入口准备完毕，设备验证阻塞。旧 SSH 别名 devenvc_yuoah.306e9d2caa0846d19bbe3625cf32cdb2.atomgit.0 无法解析；本轮未执行云端命令、CANN 编译、编译后函数体比对或 NPU 测试。没有新性能/精度 PASS 结论，没有平台提交或 git push。下一步是用实际可达连接完成上述设备闸门；不新开内核路线。
