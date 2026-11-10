# PReluGradUpdate_2329：进展与源码对照交接

## 1. 接手时状态

截至2026-11-05，本任务停止新增优化，只交接已有代码、实验与证据。平台提交和查看成绩由用户本人负责。

当前保留基线是 **J035**。用户确认官方提交 **1148578** 与J035一致，五点Pass，延迟为 **7.15、7.29、7.11、5.88、6.06 μs**。这项源码对应关系来自用户确认，未通过下载官方提交包独立核验。

J048官方反馈前三点明显回退，不能整体替代J035。J049至J052完成了隔离和测量诊断，但没有产生可保证整体收益的新默认版本。

**J053仅有本地准备，没有在云端启动。**本轮只读查询记录：

```text
TIME 2026-11-05T16:23:25+08:00
FILE J053-build.pid EXISTS False
FILE J053-build.exit EXISTS False
FILE logs/J053-build.log EXISTS False
J053_PATHS []
```

证据：本地工作目录 `J053-existing-state.log`。此结果限于已有个人云端工程根；不是对整台主机所有用户进程的检查。

## 2. 位置与版本

本地仓库：

```text
E:\suanzi\cann-ascendc-ops-collab
```

个人实验：

```text
people\JadeYara\PReluGradUpdate_2329\experiments
```

冻结基线：

```text
people\JadeYara\PReluGradUpdate_2329\experiments\J035\code
```

云端个人根：

```text
/home/developer/2329-JadeYara-20261104-1812
```

本机辅助脚本、下载检查点与分析文件：

```text
C:\Users\lin\Documents\Codex\2026-11-04\task
```

### 源码指纹

`J035-code.zip` SHA256：

```text
212183985a1795c76ca2fddf298d53ca4229321cdcdaa87517882b8ae17bbd35
```

本轮核验：ZIP内46个文件与冻结J035代码目录逐字节一致。

`J035/code/op_kernel/p_relu_grad_update_splitc.h` SHA256：

```text
f4c5f793a36a152c5ee3cd4208c359cefc3246eaacba8a930916e808bd2c1ead
```

### 必须注意本地仓库过旧

本地仓库HEAD：

```text
fc85a48701b1fb578cd8e5208c9b196a471ba267
```

它停留在P021附近，比本轮核实的GitHub main落后51个提交。本地yuhongming目录内容不足，不能据此认为该作者没有实现。

本轮固定审阅的公开仓库提交：

```text
e62693c8bd5946eebcf2ffaacb37d2bcb892bbbf
```

仓库：

https://github.com/hongwei-2026/cann-ascendc-ops-collab

只读快照：

```text
C:\Users\lin\Documents\Codex\2026-11-04\task\
  handoff_pinned_e626\
    upstream.zip
    manifest.json
    source-comparison.json
    J035_vs_hongwei.diff
    J035_vs_yuhongming.diff
    hongwei_vs_yuhongming.diff
    snapshot\
```

公开ZIP为4,178,003字节，SHA256：

```text
09eff8c110a06b28b5e9a0eab2ff2c81ab556e782f2977ab34304593f0d30422
```

ZIP CRC通过；抽取406个相关文件，逐文件指纹保存在manifest中。`handoff_review/upstream`是早先不完整的中间取件目录，不能当完整快照。

公开ZIP中未找到名称包含license的文件；不要仅因代码公开便宣称已确认可自由再许可。保留原作者、来源与变更记录。

## 3. 算子语义及设计边界

本题输入为`grads/features/weights`，输出为`dx/update`：

```text
dx     = x > 0 ? g : w * g
update = x > 0 ? 0 : x * g
```

- 通道维为`dims[1]`。
- 权重为`[1]`或`[C]`，按通道广播。
- 两个输出与输入同形。
- **update是逐元素结果，不做通道归约。**
- FP16/BF16使用FP32中间计算，再按输出类型舍入。
- `+0/-0`进入非正分支。
- Compare/Select与掩码乘法在Inf等输入上可能不同。

依据：

- J035 `samples/reference.py:47–68`。
- 固定快照 `tasks/active/PReluGradUpdate_2329/analysis/SUMMARY.md:13–30`。
- `analysis/AI-codex-claude_design_final.md:14–39、59–78`。

设计文档明确区分本题三输入两输出接口与其他同名两输入、归约输出接口，不能套用后者。

**文档与实现的限制：**设计中的“numpy验证”“位模式论证”不是NPU覆盖证明；本地有限值测试没有穷尽NaN、Inf、溢出和所有舍入边界。`reference.py`注释称与TBE一致，但设计文档明确记录了特殊值差异，应以实际表达式与测试契约为准。

## 4. 三份实际源码逐文件对照

以下作者文件均以固定提交e62693c为准。

简称：

```text
J = 本地 experiments/J035/code
H = snapshot/people/hongwei-2026/PReluGradUpdate_2329/workdir/code
Y = snapshot/people/yuhongming-2026/PReluGradUpdate_2329/workdir/code
```

| 比较 | 同文件内容 | 内容改变 | 仅左侧 | 仅右侧 |
|---|---:|---:|---:|---:|
| J035 / hongwei | 35 | 6 | 5 | 1 |
| J035 / yuhongming | 31 | 9 | 6 | 0 |
| hongwei / yuhongming | 32 | 8 | 2 | 0 |

完整逐文件SHA、差异清单在`source-comparison.json`和三个diff中。

### J035相对hongwei的不同文件

改变：

```text
op_host/p_relu_grad_update_tiling.cpp
op_kernel/p_relu_grad_update.cpp
op_kernel/p_relu_grad_update.h
op_kernel/p_relu_grad_update_tiling_data.h
op_kernel/p_relu_grad_update_tiling_key.h
tests/ut/op_host/test_p_relu_grad_update_tiling.cpp
```

J035新增：

```text
op_kernel/p_relu_grad_update_s1.h
op_kernel/p_relu_grad_update_merged.h
op_kernel/p_relu_grad_update_resident.h
op_kernel/p_relu_grad_update_splitc.h
tests/shared_weight_check.py
```

hongwei独有：

```text
tests/weight_schedule_check.py
```

### J035相对yuhongming的不同文件

改变上述六项，以及：

```text
FIX_NOTES.md
op_host/p_relu_grad_update_def.cpp
tests/local_semantic_check.py
```

J035独有四个专用头文件，以及：

```text
tests/compute_schedule_check.py
tests/shared_weight_check.py
```

InferShape、reference.py、case.json及多数工程骨架相同。因此差别主要在分派、任务组织、权重处理、转换和计算调度，不能把两份工程说成完全不同语义的算子。

### 作者版本不能混称

- hongwei当前源码是 **P105**，实现tip在仓库记录为`2e378eb`。
- P103是其历史采纳基线。
- P104平台回退，撤销。
- P105已通过仓库记录中的云端样例，**平台性能待测**。
- yuhongming的明确贡献包括P022 Compare/Select通路和P032对齐修复；未找到可独立归属于该作者当前完整实现的五点性能记录。

证据：固定快照`path-ledger.md:29–47`、`STATUS.md:5–15`。

## 5. Host分派与核内组织的真实差异

| 项目 | yuhongming当前实现 | hongwei P105 | J035 |
|---|---|---|---|
| 基本组织 | 整sample/行模式与flat模式 | flat分核，每tile不跨通道 | 普通、S1、merged、resident、split-C五族 |
| 权重 | 标量读后展开FP32向量；整行模式可复用 | 单通道标量权重，Muls | 普通路径标量；S1向量搬运；resident/split-C复用 |
| 小输入分核 | 整sample可放UB时，total也可放UB则单核 | 默认尽量满核 | shared权重按规模限核；每通道路径独立 |
| BF16转换 | 标量位展开/RNE | 标量位展开/RNE | 当前910C环境使用向量Cast |
| Select | 显式repeat参数，dx先、update后 | count接口，update先、dx后 | count接口，update先、dx后 |
| 零缓冲 | 两次清零 | 同一零缓冲复用 | 保留P103式顺序；普通路径另有features复用 |
| Workspace | 0 | 0 | 0 |
| 队列深度 | 1 | 1 | 1，专用路径另有权重队列 |
| 大通道S=1 | 可能逐元素构造权重向量 | tile受S限制，源码上会变为1 | 连续权重向量搬运及split-C |

### yuhongming：原始行模式与权重展开

`Y/op_host/p_relu_grad_update_tiling.cpp:175–190`：

- sample可以放入maxTile时，按完整sample组织。
- total可以放UB时使用单核。
- 更大输入按sample边界分核；否则走flat块。

`Y/op_kernel/p_relu_grad_update.h:159–201`：

- 整行可复用模式在Init中构造权重向量。
- 按通道段读取权重后Duplicate。
- 非32B对齐前缀先SetValue，达到8个float边界后才Duplicate。
- `innerSize=1`时可能变成大量很短的段；原设计已把这种极端情况列为限制。

`Y/...h:269–311`：

- 每通道权重用向量Mul，shared权重用Muls。
- dx Select在前，之后再次构造零向量供update。
- 使用显式repeat Select。

`Y/...h:350`的`progress += tileLen`不能机械判为跳元素bug：该实现没有P100那种按通道进一步截短currentNum的循环条件。

### hongwei P105：单通道Muls与提前准备权重

`H/op_host/p_relu_grad_update_tiling.cpp:183–191`：

- flat分核，块起点按32B元素数对齐。
- per-channel时将tileLen限制到innerSize。
- 因而S=1时按源码推导tileLen为1；这不是该输入的实测延迟。

`H/op_kernel/p_relu_grad_update.h:169–210`：

- P105把权重索引、GM标量读取、解码移到DeQue之前。
- 目的为利用CopyIn已发出后的等待窗口。
- 计算主体仍为P103式顺序，未改成J035向量权重路径。

`H/...h:254–270`：

- Muls、Mul、Duplicate准备。
- Compare(+zero)。
- update Select先读取零。
- dx Select后覆盖原零缓冲。
- 保留依赖屏障。

`H/...h:315–335`：

- currentNum受通道边界截短。
- 必须`progress += currentNum`。

### J035：增加的结构与实际切分

`J/op_host/p_relu_grad_update_tiling.cpp:161–175`：

- `weightLen>1 && innerSize==1`进入向量权重族。
- 普通路径扣除一个float缓冲预算，匹配features存储复用。

`J/...tiling.cpp:200–213`：

- shared权重采用规模相关核数。
- `weightLen>1`才认作多通道权重，避免C=1时不必要地限制成单元素tile。

`J/...tiling.cpp:216–250`：

- 根据权重周期和tile情况选择merged/resident。
- split-C只适用于S=1、C满足搬运对齐、C大于maxTile、通道块数不超过3、N至少为硬件核数两倍等条件。
- 不是按官方点号或形状名单分派。

`J/op_kernel/p_relu_grad_update.cpp:22–24、32–73`：

- 五族独立编译期选择。
- ICachePreLoad只作用于FP16/BF16普通S1入口，不是全路径预取。

`J/op_kernel/p_relu_grad_update.h:241–266`：

- features参与乘积和Compare后，存储再用于`w*g`。
- update先Select，dx后Select。
- FP16/BF16输入输出为向量Cast。

`J/op_kernel/p_relu_grad_update_splitc.h:119–125、136–163、189–211`：

- 每核固定一个通道块，遍历一段batch。
- 权重首次搬入后驻留。
- J034/J035使用256B相位调整缓冲布局。
- J035低精度每tile仍转换驻留权重；没有缓存转换后的FP32权重。

对于`[127,8192,1]`：

```text
Ctile = 4096
通道块数 = 2
batchGroups = 20
blockDim = 40
每核 = 6或7个batch
26核 × 6tile + 14核 × 7tile = 254tile
```

每核处理24,576或28,672元素；权重GM读取40次，低精度权重Cast254次。次数是源码映射推导，并经实际Block Num核对，不是硬件指令计数器。

详细依据：`experiments/J045/RESULTS.md:13–19`。

## 6. 仓库重要设计与失败教训

| 历史 | 已有记录 | 接手含义 |
|---|---|---|
| P010/P020等BF16 Cast失败 | 特定旧构建环境中后端报错 | 不把“BF16永远不可Cast”推广到已实测支持的910C环境 |
| P012只交换Select极性 | 未解决95/97错点 | 先检查mask通路，不凭约半数错误猜极性 |
| P022/P023 | Compare(+zero)，两作者Select接口不同 | 以实际编译和精度验证接口，不能混搭旧签名 |
| P019源码副本不一致 | 实际构建了旧文件 | 构建前核SHA；不能只看编辑器中代码 |
| P031/P032 | 未对齐Duplicate触发507035；修前缀后平台通过 | 保留SetValue前缀或等效已验证对齐策略 |
| P100 | 去掉未用16MB workspace、全核、单通道Muls | 原始设计不是最终性能实现 |
| P100通道截断 | `progress+=tileLen`会跳元素 | 截断后按currentNum推进 |
| P101外部模板 | 回带SyncFunc、CompareScalar、未对齐Duplicate等 | 外部“优化版”需逐项审计 |
| P102对齐DataCopy | 云样例通过，平台五点更慢，已撤回 | 指令更简单不保证平台更快 |
| P103零复用/调度 | 平台42.89；相对P100历史图接近噪声 | 不夸大历史性能增益 |
| P104常驻零缓冲 | 前三点明显更慢，后两点略快，撤回 | 额外驻留存储有代价；不能直接认定与J048同因 |
| P105提前准备标量权重 | 云端0/192错点；平台待测 | 是待验证候选，不是已确认最快版本 |

来源：固定快照`path-ledger.md:17–47`，以及`explorations/EXP001–EXP025`和`decisions/DEC001、DEC002`。

建议保留的仓库检查脚本包括：

```text
verify_codearts_source.sh
tests/local_semantic_check.py
tests/compute_schedule_check.py
tests/weight_schedule_check.py
```

这些源码/CPU检查不能替代BF16真机精度、同步压力或性能测试。

原设计推荐行模式、整核权重向量复用；yuhongming实现保留此思路。hongwei后来用P100单通道Muls替代。J035则对S=1增加连续权重搬运与split-C。因此不能说“仓库一直只有一套建议”，应按设计、实际代码和后续回退分别阅读。

## 7. 官方记录与本地测量不能混算

### 官方或仓库平台记录

| 版本/来源 | 五点μs | 得分/状态 |
|---|---|---|
| hongwei P100历史截图 | 7.26/7.41/7.33/5.72/5.82 | 42.41；截图无源码hash |
| hongwei P103 | 7.36/7.30/7.24/5.83/6.07 | 42.89 |
| hongwei P104 | 8.15/8.22/8.25/5.79/5.99 | 40.42，撤回 |
| J035，1148578，用户确认 | 7.15/7.29/7.11/5.88/6.06 | 五点Pass |
| J048，用户反馈 | 8.42/8.35/8.41/5.75/5.85 | 五点Pass；提交编号未取得 |
| hongwei P105 | 尚无平台计时 | 云端样例通过 |

不知道官方五点shape、dtype、硬件与统计协议，不能将自选`[127,8192,1]`对应某个官方点。

提交1109131的WA源码版本未知，不能称为J001。

### J035在CANN9.0的同机证据

`experiments/J035/RESULTS.md:38–50、64–109`：

- 750有限值精度案例通过。
- 27压力案例×64次，共1728次调用，逐次检查两个输出通过。
- J034/J035四轮ABBA/BAAB；每窗口250次，取101–250次中位数。
- FP32 `[127,8192,1]`：12.108→10.795μs，四轮配对差均为负。
- FP32 `[107,15424,1]`：14.801→13.152μs，四轮均改善。
- FP32 `[91,10368,1]`和`[147,6656,1]`存在单轮翻转。
- 未改低精度路径仍有顺序偏移，不支持“所有路径统一加速”。

硬件插桩第二批Compute：

```text
FP32 0.740606 → 0.560000 μs
bank group冲突比例 26.0229% → 0.2165%
bank冲突比例 0% → 1.8398%
```

插桩窗口用于机制定位，不是完整Task时间。

目标FP32普通PMU，J035：

```text
Task Duration 10.794813 μs
AIV             8.632562 μs
Scalar          4.561625 μs
Vector          2.306000 μs
MTE2            4.056312 μs
MTE3            1.955563 μs
```

流水活跃时间可重叠，不能相加，也不能把最大活跃时间直接当独占瓶颈。

### 大通道结构对照已经做过

- **J045**：固定tile/核数/UB/计算，改为连续flat通道tile并逐tile向量读取权重。包含遍历和生命周期两个相关变化。多个适用输入稳定变慢，不采用。
- **J046**：保持J035切分和遍历，仅改权重每tile重读及队列生命周期。19个适用输入普通profiler四轮全回退；无profiler6个适用输入四轮全回退，不采用。
- J046目标权重读取40→254次，逻辑GM总读写约增20.27%。不能把逻辑字节直接当实测HBM事务。
- J046 MemoryDetail采样出现接近100%的L2读命中，但限于该application replay条件，不能外推比赛缓存状态。

依据：`J045/RESULTS.md:3–19、76–102`，`J046/RESULTS.md:3–33、97–120`。

## 8. J048回退及后续诊断

J048只改split-C头文件：

1. 低精度新增独立FP32权重缓冲。
2. 首tile转换权重，后续复用转换结果。
3. 乘积仍写原neg缓冲。
4. 保留Host切分、tile、遍历、GM权重读取次数和显式同步。

官方相对J035：

```text
+1.27 / +1.06 / +1.30 / -0.13 / -0.21 μs
```

已撤销整体替代建议。没有证据证明前三点是什么dtype或shape。

可证伪解释及现有进展：

| 解释 | 已知 | 尚不能断言 |
|---|---|---|
| 新缓冲/初始化/操作数地址代价 | 真实增加UB并改变权重源地址 | 未证明它独自造成官方回退 |
| 复用次数不足 | 首tile不节省Cast，节省随每核batch增多 | 官方复用次数未知 |
| 非目标编译连带影响 | J048 FP32 split-C函数增加4字节 | 未解码为具体昂贵指令，不能对应官方前三点 |
| 测量协议差异 | 连续enqueue与普通profiling曾出现不同方向 | 无法用本地噪声“解释掉”官方回退 |
| 进程/缓存/频率状态 | 相同机器码也有跨进程计时差 | 尚未确定主因 |

J049隔离原FP32类与低精度缓存类。J050–J052在新环境进一步拆解：

```text
A/J050 = J035原源码，CANN9.3重编译
B/J051 = 新FP32权重缓冲，但每tile仍Cast
C/J052 = 与B相同布局，仅首tileCast
```

A/B不是纯初始化对照，还改变操作数地址和原地/非原地转换。B/C才更接近转换缓存条件对照。

### CANN9.3结果

- 三版本各780精度案例、57压力案例/5184次调用通过。
- 三版本完整FP32 `.o`一致。
- FP16/BF16只改变split-C函数，其他函数正文一致。
- 同进程固定分配A/A，普通profiling最大标签差0.0195μs。
- 跨进程A/A最大差1.041μs。
- `[127,8192,1]`相同FP32代码仍有C−A中位差+0.440μs，范围−0.121至+1.200μs。
- 高复用留出`[224,10368,1]`，低精度C相对B六轮都快，FP16/BF16中位差−0.795/−0.815μs；Vector均减少0.822μs。
- 相对原A仅5/6轮快，仍有+0.194/+0.170μs翻转。
- **没有采用新的按复用次数分派。**

完整27输入、轮次范围与协议在`experiments/J052/RESULTS.md:36–108`及其JSON中。

## 9. 已有实验记录如何接着读

优先顺序：

1. `J035/RESULTS.md`：当前保留版本、布局证据和局限。
2. `J038/RESULTS.md`：零向量常驻，局部Compute改善不等于全局稳定收益。
3. `J045/RESULTS.md`：连续flat结构对照。
4. `J046/RESULTS.md`：严格权重生命周期对照。
5. `J048/OFFICIAL-REVIEW.md`：平台回退、逐项变更与回滚。
6. `J049/RESULTS.md`及ELF记录：编译隔离。
7. `J050/RESULTS.md`、`J052/RESULTS.md`：升级、A/A、A/B/C。
8. J027–J032：转换复用、预取、队列、依赖等待和模拟器相关历史，避免重复把仿真改善当实板收益。
9. 固定仓库`path-ledger.md`、EXP001–025、DEC001/DEC002：作者原始失败和回退。

较早单轮或旧协议收益应保留原环境、shape、dtype和统计方法，不跨实验直接排名。

## 10. 回滚与恢复

J048仅替换一个头文件且没有其他编辑时，可由接手人取回：

```text
E:\suanzi\cann-ascendc-ops-collab\people\JadeYara\
PReluGradUpdate_2329\experiments\J048\manual-platform-trial\
rollback\op_kernel\p_relu_grad_update_splitc.h
```

该文件与冻结J035头文件、ZIP内文件一致，SHA见第2节。

如果其他文件也改过，应从完整J035冻结包在新目录恢复，并核对46个文件，不应只换一个头文件便宣称整版恢复。

恢复本地/编辑器源码不会自动恢复官方成绩。官方提交和结果查看由用户本人完成，本交接没有执行该动作。

### 关键归档

J035完整证据ZIP：

```text
J035-910C-evidence.zip
179,054,045 bytes
SHA256:
4f97f7a2e3d023f21a3962af4a55dbac4c97b7ab78f6abdc8fc76095042850f4
```

J050–J052完整证据：

```text
E:\suanzi\cann-ascendc-ops-collab\tasks\active\
PReluGradUpdate_2329\explorations\logs\
EXP_J050_J052_CANN93_ABC_20261105\
J050-J052-910C-evidence.zip
```

```text
459,137,010 bytes
SHA256:
ec5032a50c72932b59039a3216c814bf1715413e047f4dffd0efd9160e50e4ac
```

已完成CRC、二进制路径、blob引用、依赖归档和普通文件SHA核验。原始profiling不应只靠摘要替代。

J053本地源码包：

```text
J053-code.zip
SHA256:
7c4a50ebaa7efbf024cb7c3bcd0de1ea7b148e3b3233204785c05ada290a4ca4
```

它是测试专用A/B/C算子别名准备包，**未构建、未验证，不是提交候选**。

## 11. 接手人必须知道的缺口

- 官方五点shape、dtype、环境和统计协议未知。
- 未取得J048官方提交编号和实际上传快照。
- 未取得J048之后恢复J035的官方控制结果。
- J053同进程多实现对照未完成。
- 新环境为9.3安装目录，但compiler/OPP明确标记beta.1，不应省略。
- 原始设计中的910B限制不能直接当910C结论。
- 910C/A3工具不支持的核内上板指令timeline，不能以模拟器时序冒充。
- 自选大通道的10–20μs不证明官方7μs整体变差，也不能证明已达到官方4–5μs目标。
- 新优化方向尚未决定。本交接不授权自动恢复实验、提交或推送。
