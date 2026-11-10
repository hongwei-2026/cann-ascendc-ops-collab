# PReluGradUpdate_2329 当前进度与 DeepSeek 调研交接

更新：2026-11-08。本文接续 11 月 7 日交接稿，状态已更新至 J072 结案：不晋升，保留 J071。

## 一、现在的结论

1. 最新可复核、可交给用户试测的独立标准候选是 **J071**。它包含 J069 连续 Gather、J070 泛化合同支持，以及 J071 真 TBE 数值链下的 FP32 split-C 双槽流水。
2. J071 不是新的生产默认版。冻结生产仍是 J035，J070/J071 都独立保存，未因本次交接修改源码、运行设备测试或提交平台。
3. J071 的明确本地收益：FP32 [127,8192,1]，J070 13.6654 → J071 11.8605 μs，快13.21%；11个启用输入66/66轮达成 >0.1μs 且 >2% 收益，762/762严格控制通过。
4. 最大未解决问题是 General 泛化路径：J070测的12个新输入中，仅互补广播FP16/BF16快6–8%，其余比内置TBE慢22.5–170.1%。J071没有优化这些路径。
5. J072 General三输入批量准备已结案、证据已校验：窗口边界快6.6–7.0%，互补广播慢1.8–2.2%，15个BF16旧输入退化、最差21.55%。不晋升，仍提交已验证的J071，不混入J072。
6. 官方五点 shape、dtype 和完整计时协议未知。不得把本地输入与官方点号绑定，也不得按耗时猜隐藏shape。

## 二、这次提交哪些文件

### 2.1 推荐以完整 J071 标准包为唯一来源

完整路径：
C:\Users\lin\Documents\Codex\2026-11-05\task-2\publish-repo\people\JadeYara\notes\PReluGradUpdate_2329\experiments\J071\J071-standard-code.zip

包顶层为 code/，共52个文件。
SHA256：34e7099fc0d84471b9fb78894cab2315216dc8bd077f07e80db46964e33ef85f

本次已重新校验包SHA、ZIP CRC和52个条目与 J071/candidate/code 的逐字节对应，全部一致。方便取用的同字节副本位于：E:\suanzi\J071-standard-code.zip。

若平台支持整工程上传，使用这个完整标准包。若平台是逐文件编辑，按下一节从同一个包同步，保持原工程相对目录。不要上传 abc/ 别名测试工程、device-evidence.zip、source.zip 实验包、.part 文件或任意旧 unvalidated 包。官方操作仍由用户本人完成。

### 2.2 相对“完整 J069 标准源码”的准确增量

基准包：J069/candidate-standard-source.zip，48文件，SHA256：
7c7bf2954376f0ae7bb34cebe1e4faa27f9291004d750bacb40562ff26bb60ae

这是本地J069→J071的SHA差异，尚未逐文件读取当前平台编辑器；只有平台确实为完整J069时，才可把下面9项当最小增量。平台若混有J065/J070或手工编辑，请以完整J071重建一致版本，避免入口与头文件不同代。

**替换5项（相对 code/）：**
- op_host/p_relu_grad_update_def.cpp
- op_host/p_relu_grad_update_infershape.cpp
- op_host/p_relu_grad_update_tiling.cpp
- op_kernel/p_relu_grad_update.cpp
- op_kernel/p_relu_grad_update_tiling_key.h

**新增4项：**
- op_host/p_relu_grad_update_shape_utils.h
- op_kernel/p_relu_grad_update_general.h
- op_kernel/p_relu_grad_update_general_tiling.h
- op_kernel/p_relu_grad_update_full_pipeline.h

**删除0项；其余43项完全相同。**

请特别保留已有 p_relu_grad_update_gather.h、gather_tiling.h、tiling_data.h 及原各路径头文件，入口仍会引用。不要只换一个 p_relu_grad_update.cpp。

如果平台已经是最终实测J070 attempt5，则 J071 本身只替换 op_kernel/p_relu_grad_update.cpp，新增 op_kernel/p_relu_grad_update_full_pipeline.h。不要把这一“2文件增量”错误用于J069。

## 三、目录与版本入口

最新工程 R：
C:\Users\lin\Documents\Codex\2026-11-05\task-2\publish-repo

最新个人笔记 N：
R\people\JadeYara\notes\PReluGradUpdate_2329

实验根 X：
N\experiments

冻结生产：
R\people\JadeYara\PReluGradUpdate_2329\workdir\code

E:\suanzi\cann-ascendc-ops-collab 是旧副本，不能据其旧README恢复最新实验。本文放到E盘是为了交接方便，不表示最新工程迁回E盘。工作树含未提交实验，不执行 reset/clean，不从旧副本覆盖最新工程。

最新状态优先看 X/J071/README.md、X/J072/README.md、N/path-ledger.md、对应DEC和实测证据。N/optimization-status.md及共享SUMMARY包含历史状态，不能覆盖J067–J071新发现。共享STATUS还是另一位Owner的P105记录，不等于JadeYara最新进度。

## 四、官方与本地成绩分开看

| 版本 | 官方记录 | 五点 μs | 说明 |
|---|---|---|---|
| J035 | 1148578，五点Pass | 7.15 / 7.29 / 7.11 / 5.88 / 6.06 | 本地checkpoint记录用户确认，冻结生产 |
| J056 | 1191764，五点Pass | 7.21 / 7.16 / 7.19 / 5.58 / 5.84 | 后续J057记录确认，独立候选 |
| J069 | 用户截图复核：五点Pass，页面0.00% | 7.55 / 7.43 / 7.56 / 5.77 / 5.83 | 未取得提交ID，不猜测；截图来源用户消息 Sentinel_5e2a34145788819181b3956c932546a9 |
| J070/J071 | 尚无已核验官方成绩 | 无 | 本地通过不等于官方收益 |

J069前三点比J035慢，后两点快，不能表述成五点整体突破。截图与本地记录不是平台提交快照的源码哈希证明；本次没有登录平台重新验证文件列表。J063曾由用户报告官方无变化；J065发生过旧入口6参与新头7参混合导致编译错误，随后同步记录说明入口配套问题。不要再混搭版本。

下面的本地数据来自原910C / CANN9.3 / ascend910_93，指标为设备完整Task Duration，不是CPU端到端，也不是官方分数。

## 五、最重要的语义纠正：真 TBE 不是简单 where

旧共享SUMMARY及11月7日交接稿按NumPy where写作：
dx = x>0 ? g : w*g；update = x>0 ? 0 : x*g。

J067实板与CANN9.3源码证实：合法有限输入的溢出、Inf/NaN传播下，这与真实TBE不等价。J068起保留实际TBE执行链，不能再用旧where优化替换。

当前链路：
- neg = select(x>0, 0, 1)，pos = 1-neg。
- dx_compute = (pos + neg*w)*g。
- dx输出量化到原dtype；FP16/BF16再将已量化dx转回FP32。
- update = (neg*x)*g + quantized_dx*0，再量化回输出dtype。
- dx*0不能当无用项删掉：Inf*0/NaN会影响update；不能擅用融合乘加、原生half或条件where代替。
- update是逐元素输出，不做通道归约，不用atomic。

真实格式/广播合同：
- 输出shape为 broadcast(grads, features)，weights只广播到该输出，不能扩大输出。
- weights映射必须遵循TBE源码分支顺序，包括rank(features)-1空间权重；reshape不能改变实际存储元素总数。
- ND、NCHW与物理NC1HWC0；三输入/两输出dtype一致，BF16独立处理。
- NC1按实际[N,C1,H,W,C0]物理存储。例如逻辑C=19实际权重有32个lane，不能按19取模，也不能擅做layout转换。
- GE会将外部ND权重内部标作NCHW，同时存储仍是[7]；最终Host已兼容这一连续存储标签。
- NC1格式带子格式位，比较应取SDK主格式，不能直接拿原始整数当简单枚举。
- rank8为已核实TBE检查上限；不自行发明grads最低rank=2。标量/rank1边缘组合以实际TBE是否接受为准。
- 空输出用合法General key与一块no-op，在任何除数、suffix和GM访问前退出。
- 未知rank图编译仍未验证；已测描述通过不代表所有动态图、混合格式转换均支持。

实际源码：X/J067/reference-evidence/prelu_grad_update.py。
行号索引：X/J070/reference-lines.json。重点：op_select_format 34–95、比较/选择98–150、compute155–219、权重映射222–255、入口261–343。

旧golden差异必须分账，不能覆盖旧文件来制造通过。比较器须说明有限值、±0和NaN类别策略，不宣称NaN payload逐位完全一致。

## 六、J069：连续 Gather 已证明结构性收益

从真TBE数值基线J068出发，用连续tile、紧凑权重窗口和有界整数Gather处理原ND/same-g-x/一维通道权重路径；不包含J070通用格式/广播fallback。

- 1063组：970修正legacy + 27实际TBE oracle + 57相位/窗口 + 9新增TBE特殊值。
- ABC4584调用、标准1528调用全部通过；标准/B54/54、A/C45/45函数对应通过。
- 127输入，3048窗口，1,066,800调用。
- 53个Gather输入318/318轮快 >0.1μs且 >2%；输入级改善中位86.135%，范围38.593–99.202%。比较对象是低效但数值正确的J068，不能套到J035或官方。
- 74个未改输入444/444严格四路控制稳定；整体严格控制753/762、A/A757/762，9个异常完整保留。
- FP32 [8,64,3] 11.48025→6.20025μs；FP32 [127,1024,3] 1320.21875→12.161μs。

已踩坑：ShiftRight的int32移位参数、REGISTER_TILING的带引号TILING_KEY_VAR表达式、UI_LIST编码序号不等于schMode。不得把24/25/26直接当所有构建组合的编码key。中止运行、容量满和早期失败日志都保留。

证据：X/J069/DEC-J069.md、performance-interpretation.json、device-evidence.zip。

## 七、J070：泛化正确性基线，性能仍弱

实现General小窗口Gather：固定tile上限2048，同时按最小输出suffix周期切分；每个来源取一个精确字节连续GM窗口，在UB内Gather。绝对offset/shape/suffix为uint64，只将已证明有界的局部差转为int32。尾部索引先夹到n-1，读写不越实际存储。

General独立760B元数据，不能reinterpret旧80B或Gather88B结构。最终编译描述opParaSize为768。General schMode28/29/30；编码序号随dtype构建组合变化。

四项集成修复：InferDataType注册、Old80/Gather88/General760 ABI、NCHW内部权重标签、NC1子格式位。最终包必须用 visible-agent/attempt5，不能回退初始candidate/unvalidated源码。

验证：
- 63个提议，TBE接受61；标准/B全部通过、逻辑差异0。拒绝的probe_scalar_grads与probe_rank1_features_scalar_weight不计通过。
- FP16与BF16各覆盖65536种feature位型的广播输入，两算子均通过。
- 原1063组：ABC4584调用、标准1528调用全部通过。
- 标准/B189/189函数同字节；A/C54/54；B/A旧函数39/54相同，不同者实际纳入回归。
- 旧路径127输入性能，基线762/762稳定；四路同速757/762。5个例外均为候选略快越2%同速阈值，没有删除重测。输入级变化中位快0.0081%，基本持平。

General对内置TBE：12输入、72轮，基线控制72/72。
| 输入 | TBE μs | J070 μs |
|---|---:|---:|
| ND互补FP32 | 7.5326 | 9.2305 |
| ND互补FP16 | 11.4750 | 10.7400 |
| ND互补BF16 | 11.3001 | 10.4400 |
| ND窗口边界FP32 | 14.8504 | 27.8407 |
| ND窗口边界FP16 | 15.3201 | 30.0053 |
| ND窗口边界BF16 | 15.3003 | 29.9005 |
| NCHW通道FP32 | 4.3603 | 8.3400 |
| NCHW通道FP16 | 4.7527 | 8.8400 |
| NCHW通道BF16 | 4.7576 | 8.8400 |
| NC1 C19广播g FP32 | 3.9075 | 10.5553 |
| NC1 C19广播g FP16 | 4.4150 | 9.9352 |
| NC1 C19广播g BF16 | 4.4200 | 8.9153 |

这不是普遍达到TBE95%性能要求的实现。来源：X/J070/visible-agent/RESULT.md。

## 八、J071：把已证实流水迁到真TBE链

仅在原FP32 split-C且tileLen=4096启用输入/输出双槽。Compute逐字保持J070真TBE链，不移植J059旧where公式；Host、核数、tile、权重驻留和其他路径不改，不叠加J060队列合并。

- UB185088B / 可用188416B，余3328B；保留256B相位，两槽不翻转相位。不能直接再加缓冲。
- 独立firstInput，权重只加载一次；末批不越界预取。
- 67个公开ACL提议，TBE接受65个，标准/B全过、逻辑差异0。
- 4个新split-C：[81,8128,1]、[127,8192,1]、[127,12104,1]、[257,8192,1]，真实TBE golden含NaN/Inf、有限溢出、±0、正规/非正规，覆盖尾通道、不同批数及39/40核。
- 1067组：ABC4776逐调用检查、标准1592，全部通过。新增4组每接口重复16次。
- 127输入、1,066,800调用、762轮；ABCA/CAAB/BACA/ACBA/BAAC/CABA，同进程/stream/五张量地址；每窗350次弃前200，后150中位，窗末核对两输出。
- 11启用输入66/66轮收益达门槛；A/A762/762、严格控制762/762。
- 目标[127,8192,1] 13.6654→11.8605μs；11输入中位改善范围约8.73–17.80%。
- B/A FP16与BF16各63/63函数体一致，FP3248/63；C/A与标准/B各三dtype63/63一致。A与此前实测J070189函数一致。
- 初次probe执行位错误、输入hash审计缺字段均保留：campaign/campaign-r2失败，campaign-r3成功。不能删除失败或称每次启动均成功；性能只完成一次，不靠重跑刷门槛。

证据：X/J071/README.md、DEC-J071.md、source-check.json、pipeline.diff、evidence/result-summary.json、evidence/function-bodies.json、evidence/baseline-vs-measured-J070.json。
J071未重做仿真时间线，J059的旧链时间线只能作为立项依据，不能充作当前实板收益归因。

## 八点一、J072最新结案：不晋升，保留J071

本交接编写期间J072完成归档，已重新读取最终README与evidence-meta，不再按进行中处理。

唯一变化：General三个来源各自compact/raw缓冲，集中搬运、一次MTE2/V完成屏障，之后仍原Cast/Gather。Window、整数索引、真TBE Compute保持，tile2048与分核保持，Host只调整UB核算。FP32/低精度UB为92672/113152B。

- 构建通过；69个ACL提议TBE接受67，标准/B全过、逻辑差异0。复用已核验hash的真实TBE oracle，不声称本轮重新生成全部golden。
- 1067回归：ABC4776次、标准1592次，逐调用双输出全过。
- General12输入：7个输入级中位改善，3个同时超过0.1μs与2%；获胜41/72轮，严格获益18/72轮。J071基线控制71/72，内置TBE控制72/72。
- 窗口边界FP32/FP16/BF16相对J071分别快6.75%/7.00%/6.59%，但仍比同轮TBE慢69.61%/81.50%/81.17%。
- 互补广播三dtype分别慢2.22%/1.97%/1.83%；低精度虽仍快于同轮TBE，也不等于比J071改善。
- 旧路径127输入严格控制672/762；15个BF16输入共90轮失败，异常轮的A/A自身控制90/90通过，不能解释为已证实的基线漂移。
- 最差BF16 [127,64,1] wlen64：5.972625→7.265250μs，慢21.55%。根因未定位，现有数据足以拒绝晋升。
- 标准/B189函数全部一致；A与上轮实测J071189/189。B/A FP32与BF16各60/63、FP16为45/63；旧BF16路径函数体相同并不能证明整个包布局/元数据/资源配置相同。
- J07152源码与J03546源码保持不变。一次性能测量，失败保留，不追加候选。

完整结论：X/J072/README.md；逐例结果：X/J072/evidence/result-summary.json；编译对应：X/J072/evidence/function-bodies.json。
证据包X/J072/device-evidence.zip为65124628B、1418文件，SHA256：523ea64f509999793273a471e2918d705ef45788c3d97c7d3764c55427e383ed；最终校验记录确认整包/每文件SHA、大小、CRC及标准/ABC云测源对应。J072标准包仅为失败候选留档，不替代本次J071提交包。

## 九、不要无证据重开的路线

- J005–J007/J028全面双缓冲：没有普遍稳定完整延迟收益。J071只证明特定FP32 split-C4096分派，不能泛化全路径。
- J046每tile重读权重：已有对照更慢，不能以GM→UB搬运就推断HBM瓶颈。
- J048权重缓存：本地信号未转成官方整体收益，前三点退化；布局和转换成本需要隔离。
- J054紧凑mask/复用/两行：目标FP32比J035慢，不采纳。
- J060队列合并：局部目标快，但控制151/168，未改BF16回退，不直接叠到J071。
- J061/J062/J063 ShortS：目标收益但控制失败；J062251/270，J063257/270。
- J064恢复BF16原函数同字节仍回退：194/270控制。函数相同不等于运行状态/完整延迟完全相同，不能断言唯一原因已找到。
- J065 TPipe外置：本地270/270轮改善、267/270控制，但官方前三点回退；不可据本地单轮替代官方证据。
- J066组合：精度通过，性能主动中止，无完整性能结论。
- J067原生half：旧reference数值与性能证据有效，但不能称真TBE全合规；真TBE已证溢出传播不同。
- J068真TBE修正：作为数值基线重要，性能回退不允许通过退回错误where语义“解决”。
- 未对齐Duplicate曾有507035；归约/atomic和GM权重展开不属于当前候选路线。

## 十、请 DeepSeek 重点调研什么

请先阅读本文及当前源码，不按官方五个点反推shape，也不要直接重写整套算子。期望输出：按优先级排列的2–3个可证伪方案，每个给源码位置、性能机制、数值/格式风险、最小A/B、通过/停止标准和公开技术依据。

1. **优先解释J072失败，不重复已有实验。** 三来源批量DMA只对窗口边界快6.6–7.0%，互补广播退化，且15个BF16旧路径稳定回退。请提出能区分包级布局/资源配置、指令与数据依赖、运行状态等假设的最小只改一项实验；不能预设唯一根因，也不能重测刷掉失败。
2. **分析General固有工作量。** J072已证明单纯减少源间等待不足以解决整体问题，检查索引生成、Gather与重复格式/坐标计算是否主导完整Task；区分恒等、标量、一维通道、少量广播轴等合法结构，提出最小分派，而不是按样例shape硬编码。
3. **探索真TBE链下低精度优化。** FP16/BF16转换、dx量化后回读及dx*0依赖是合同部分。任何简化都先说明在哪个严格定义的数据域等价，并用实际TBE特殊值/全位型验证，不沿用旧where证明。
4. **评估流水扩展边界，而非盲扫参数。** J071UB只余3328B；分析非4096、不同复用/尾块和低精度能否受益，先给可行UB/依赖图，再单候选验证。不要把J059/J071结果当所有shape充分证据。
5. **官方收益原因仍开放。** 本地910C/9.3与官方环境未知，先核对提交文件一致性和编译目标，拿到实际公开配置再提出解释。不能宣称官方前三点必然是哪条路径。

验收沿用：实际TBE独立oracle、标准与别名B均验证、防内置fallback、源码/编译函数对应、逐调用输出与特殊值、同进程完整Task、平衡顺序、未改路径控制。收益 >0.1μs且 >2%；控制 ≤0.1μs且 ≤2%。失败轮保留，不降低门槛、不选择性补跑。

本文件是调研交接，不授权DeepSeek访问账号、上传私人数据、改生产、提交平台、开新付费资源或公开push。后续执行范围由用户另行决定。

## 十一、源码与证据索引

以下均相对实验根 X；首次使用时将第三节X展开为绝对路径。

- J069/candidate-standard-source.zip：48文件；真实TBE连续Gather标准包。
- J069/device-evidence.zip：37424049B，581文件；SHA256 6210da2d5f8fdce1d45e4f1ab719dfdcca2c1d227728242bac9995a64a137b83。
- J070/visible-agent/attempt5/candidate-standard-source-attempt5.zip：51文件；SHA256 9b8320429169af39ffd1529a35a6418ad9a21130855222793d705c7fa79b200e。
- J070/visible-agent/device-evidence.zip：60544364B，1432文件；SHA256 ab5ce45569d476fc69664dd34d5d456b6c3b85392f87c2bc71c53f5f0051d0f4。
- J071/J071-standard-code.zip：本次交付52文件标准包，见第二节SHA。
- J071/device-evidence.zip：54135153B，1231文件；SHA256 ae0a8e15b959382d7da5ab89508dff1aca54aa224acb4938252a945ffd9d8f4e。
- J071/candidate/code/op_host/：最终Host；op_kernel/p_relu_grad_update.cpp入口；full_pipeline.h为新增流水；general.h为泛化路径；gather.h为J069连续Gather。
- J071/evidence-meta.json、J070/visible-agent/archive-meta.json：证据校验与范围。
- 云端历史根 /mnt/workspace/2329-JadeYara-20261104-1812；J070/attempt3-visible和J071为对应目录。本次未重新访问远端，具体归档范围按包内ARCHIVE_SCOPE.json。部分原始profile数据库/时间线和继承输入仍在云端，不能声称所有原始大文件都在本地ZIP。

交给DeepSeek的最小材料：本文 + J071-standard-code.zip。若它需要复核具体性能，再补 J070 RESULT、J071 README/result-summary 与相关证据；不要默认把全部历史大包上传。提交平台与交给DeepSeek是两件事，本文件和证据包不属于算子提交源码。

