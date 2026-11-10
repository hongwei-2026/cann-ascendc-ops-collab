# PReluGradUpdate 2329 大通道优化审阅：暂停于 J041 启动前

截至 2026-11-05。本文件只汇总已有记录；没有启动新增设备作业、修改算法或实施“小通道性能换大通道提速”的取舍。

## 当前状态

- **J040 已结束并归档**：布局/精度与性能作业退出码均为 0。最后只读远端检查时间为 2026-11-05 08:52:35 +08:00，NPU 运行进程数为 0。这是该时刻的观测，不表示持续监控。
- **J041 尚未启动**：只完成本地 48 组小数据量选点与脚本准备；没有上传/启动新设备测试、没有新算法。准备文件保留并标记暂停，不能把它们当作测试结果。
- 保留 J035 冻结基线、J038 增量候选，以及 J039/J040 实验证据。J039/J040 尚未晋升，未覆盖或替换基线。
- 没有比赛提交、代码推送、网络设置变更；现在等待用户审阅后决定方向。

## 如何读数字

形状统一记为 **[N,C,S]**，S 为空间维元素积。下列 S=1 向量权重样本均为 weightLen=C。`[127,8192,1]` 是大通道；`[600000,1,1]` 是大总量、单通道；两者不能混称。C=256/512 的早期实验也不代表 C=8192 的路径。

设备记录为 Ascend910_9362 / ascend910_93 / dav-c220-vec，CANN 9.0.0，driver 25.5.5。表中单位为 μs，指标为 msprof **Task Duration(us)**，不是 Python/主机往返耗时，也不是 aiv_time 或模拟器周期。

每一行只比较同一实验、同一输入、同一采样方式的基线与候选。四轮统计为每轮统计量的均值；ABBA/BAAB 中，每轮先平均同版本两个位置的采样中位数。四轮范围不是置信区间，也不是全部原始调用的最大最小值。早期 25 次调用/舍弃前 5 次，与后期 250 次/取第 101–250 次，以及单独运行与批量运行，不能直接拼接成累计收益。

榜单五点的输入未知，现有本地 10–13 μs 大通道样本不能对应榜单 4–5 μs 目标，更不能据此宣称达到或无法达到目标。

## 大通道主线：同一目标输入的各轮实测

本表全部为 **[127,8192,1]，weightLen=8192**。负差值表示候选更快。四轮全快仅表示本次四轮方向一致；仍须结合未修改控制路径判断归因。

| 实验/基线→候选 | dtype | 基线→候选 μs | 差值 μs | 四轮差值 μs | 四轮全快 |
|---|---|---:|---:|---|---|
| J024-hot：J023→J024 | BF16 | 17.7845 → 13.0721 | -4.7124 | -4.5600, -4.5410, -5.1190, -4.6295 | 是 |
| J024-hot：J023→J024 | FP16 | 17.6998 → 12.9396 | -4.7601 | -5.3590, -4.3410, -4.6805, -4.6600 | 是 |
| J024-hot：J023→J024 | FP32 | 14.1798 → 11.8945 | -2.2852 | -3.0010, -2.1790, -1.7410, -2.2200 | 是 |
| J026-validation：J023→J026 | BF16 | 17.8574 → 13.2950 | -4.5624 | -4.5495, -3.6400, -4.1600, -5.9000 | 是 |
| J026-validation：J023→J026 | FP16 | 17.7818 → 13.1124 | -4.6694 | -4.3995, -3.8600, -4.3390, -6.0790 | 是 |
| J026-validation：J023→J026 | FP32 | 13.9246 → 12.1796 | -1.7450 | -1.9405, -0.9100, -1.2600, -2.8695 | 是 |
| J027：J026→J027 | BF16 | 13.1371 → 13.2649 | +0.1278 | +0.1800, -0.3005, +0.3305, +0.3010 | 否 |
| J027：J026→J027 | FP16 | 12.8721 → 13.4299 | +0.5577 | +0.6000, -0.3600, +0.9100, +1.0810 | 否 |
| J027：J026→J027 | FP32 | 12.0071 → 12.5750 | +0.5679 | +0.5400, +0.0505, +0.8405, +0.8405 | 否 |
| J029：J026→J029 | BF16 | 13.3495 → 13.3609 | +0.0114 | -0.0905, -0.0648, +0.2605, -0.0598 | 否 |
| J029：J026→J029 | FP16 | 13.2572 → 13.2485 | -0.0087 | -0.1798, -0.0200, +0.3652, -0.2005 | 否 |
| J029：J026→J029 | FP32 | 12.2099 → 12.1921 | -0.0177 | -0.1100, -0.1210, +0.2000, -0.0400 | 否 |
| J032：J026→J032 | BF16 | 13.0621 → 13.0836 | +0.0214 | +0.0205, +0.4552, -0.3595, -0.0305 | 否 |
| J032：J026→J032 | FP16 | 12.8873 → 12.7973 | -0.0901 | +0.0297, +0.2895, -0.3347, -0.3447 | 否 |
| J032：J026→J032 | FP32 | 11.9398 → 11.9572 | +0.0174 | +0.0105, +0.4395, -0.3850, +0.0045 | 否 |
| J034：J026→J034 | BF16 | 13.2649 → 12.0236 | -1.2413 | -0.4752, -1.9400, -0.8900, -1.6600 | 是 |
| J034：J026→J034 | FP16 | 13.2048 → 11.7572 | -1.4476 | -0.5795, -2.3205, -1.2508, -1.6395 | 是 |
| J034：J026→J034 | FP32 | 12.1073 → 12.0810 | -0.0263 | +0.7898, -0.7200, +0.2905, -0.4652 | 否 |
| J035：J034→J035 | BF16 | 12.0974 → 11.9374 | -0.1600 | +0.7700, -0.6800, -0.1100, -0.6200 | 否 |
| J035：J034→J035 | FP16 | 11.8200 → 11.8324 | +0.0124 | +0.8400, -0.4200, +0.2195, -0.5900 | 否 |
| J035：J034→J035 | FP32 | 12.1082 → 10.7948 | -1.3134 | -0.4243, -1.8397, -1.2595, -1.7300 | 是 |
| J038：J035→J038 | BF16 | 12.0461 → 11.8396 | -0.2064 | -0.3448, -0.1905, +0.0090, -0.2995 | 否 |
| J038：J035→J038 | FP16 | 11.9283 → 11.6397 | -0.2886 | -0.3300, -0.7995, +0.0105, -0.0352 | 否 |
| J038：J035→J038 | FP32 | 10.9124 → 10.8386 | -0.0738 | -0.1245, -0.1200, +0.1500, -0.2005 | 否 |
| J039：J038→J039 | BF16 | 11.9025 → 11.8547 | -0.0478 | -0.0800, -0.1505, +0.1695, -0.1303 | 否 |
| J039：J038→J039 | FP16 | 11.8336 → 11.7724 | -0.0612 | -0.0547, +0.0405, -0.0400, -0.1905 | 否 |
| J039：J038→J039 | FP32 | 10.8862 → 10.8123 | -0.0739 | -0.2102, -0.1402, +0.2048, -0.1500 | 否 |
| J040：J038→J040 | BF16 | 12.0399 → 11.6849 | -0.3550 | -0.0205, -0.5800, -0.6100, -0.2095 | 是 |
| J040：J038→J040 | FP16 | 11.8048 → 11.5723 | -0.2324 | -0.0100, -0.4802, -0.3695, -0.0700 | 是 |
| J040：J038→J040 | FP32 | 10.9399 → 10.6111 | -0.3287 | +0.0700, -0.5602, -0.5647, -0.2600 | 否 |

J024-hot 取第101–250次；J026/J027取第101–150次，其余本表实验取第101–250次。J029及之后采用四轮 ABBA/BAAB。完整形状、四轮统计和数据源 SHA256 在旁边的 `LARGE_CHANNEL_REVIEW_DATA.json/csv`，包括未修改控制路径，未删掉不利点。

### 可以归因的主要收益

1. **J024/J026：按通道块遍历 batch，复用权重。** C=8192 超过原驻留路径的最大 tile（FP32 5184、低精度4288），不能直接套“整个C驻留”。split-C 把 C 分成两块，40核、20组batch，每核处理6/7个batch；目标权重逻辑读取量减少84.25%，总逻辑GM流量减少16.85%。J024相对J023三dtype四轮全快，MTE2/MTE3随之下降。流量是代码逻辑估算，不是实测HBM字节数。
2. **J034/J035：调整 UB 地址布局。** J034低精度、J035 FP32只改split-C地址分配；J034目标FP16/BF16分别快1.448/1.241 μs，J035目标FP32快1.313 μs，均四轮同向。实板分段Compute也下降：FP16 1548→1264周期；FP32 1222→924周期。银行组冲突指标与变化一致。这比单看流水活跃占比提供了更直接的证据。
3. **J038：零向量只初始化一次。** 相对J035目标三dtype均值改善，但各有一轮反向；28个真正改动组合中17个四轮全快。第二次Compute分段FP32 924→857–858、FP16 1264→1163、BF16 1340→1227周期。保留为有依据的增量候选，未替换冻结J035。

### 已验证的适用范围与反例

| 同批比较 | 输入 / dtype | 基线→候选 μs | 解释 |
|---|---|---:|---|
| J023→J024，250/101–250 | [31,32768,1] FP16 | 12.410→13.310 | 四轮均回退；不能将split-C扩到全部大C |
| 同上 | [31,32768,1] BF16 | 12.707→13.422 | 四轮均回退 |
| J023→J026，150/101–150 | [103,9728,1] FP16 / BF16 | 15.870→12.510 / 15.970→12.737 | 留出形状也有收益 |
| 同上 | [147,6656,1] FP16 / BF16 | 19.322→13.160 / 19.404→13.292 | 并非只对8192通道优化 |
| J035→J038，250/101–250 | [113,12864,1] FP32 / FP16 / BF16 | 12.532→12.225 / 13.885→13.481 / 14.130→13.781 | 三dtype四轮全快的新留出点 |

J026采用已有对齐/S=1/向量权重条件，再限制 `ceil(C/maxTile)<=3`、`N>=2*hardwareCoreNum`（本机40核），两通道块40核、三块39核；是经留出验证的经验规则，不是具体shape查表，也不是所有输入最优的保证。J025更复杂选择规则在留出集出现11个四轮全败，已放弃。

## 从早期版本到当前：策略、结果和失败边界

下表数值都应在各自报告内解读；“未证明”与“已测回退”分开。PReluGradUpdate两个输出是逐元素计算，本任务没有跨通道归约。

| 版本 | 改动/用途 | 已有结果或停止原因 | 对小通道/普通路径的关系 |
|---|---|---|---|
| P103/J001/J002 | J001仅将Host两个weight判断改为weightLen>1；J002验证基础设施 | P103/J001各54精度通过；C>1算子本体不变 | 修正C=1标量路径选择；非大C新算法 |
| J003 | BF16向量Cast替代旧桥接 | 54精度通过；[600000,1,1] BF16旧约673 μs降至9.28–10.07 | [2,3,65] BF16约6.03–6.31降至4.94–4.99；大总量不等于大C |
| J004 | S=1连续权重向量Mul代替逐通道Muls | 96通过；[127,256,1] FP32两轮307.803/307.353→10.239/9.759；BF16 343.833/343.253→12.320/12.360；[129,257,1] FP16 360.262/361.173→13.360/13.860 | 普通[2,3,65] BF16两轮4.760/4.630→5.260/4.910，回退0.50/0.28 |
| J005/J006 | 模板分派、Host独立tiling key隔离 | J005 96、J006 45精度通过；普通路径方向翻转，S=1无稳定增益 | 未消除全部回退，保留J003和专项J004 |
| J007 | 输入双缓冲、下一tile预取 | 96通过；[600000,1,1] FP32 9.180/9.160→9.480/9.889；低精度翻转，无稳定收益 | UB预算上升、tile可能缩小；不能单凭重叠设想采用 |
| J008 | 复用featuresFloat为负分支，删中间缓冲 | 96通过；同[600000,1,1] FP32 9.2595/9.220→8.660/8.709 | 普通路径适用；低精度收益方向不一致 |
| J009 | 缓存标量权重及索引 | 102通过；FP32目标相对J008一轮退一轮进，未证实新增收益 | 不作为通用替换 |
| J010/J011/J012 | 单权重按规模分核，最少核数对照 | J011 240精度、90组阈值网格；86组两轮改善≥0.15；J012通常较慢 | 主要改善小输入；weightLen>1不走此规则 |
| J013 | 整合J011分核、J004 S=1、J008 UB复用 | 273通过；[127,256,1] FP32 J004 9.630/9.479→J013 9.680/9.320；BF16 11.700/11.199→11.720/11.430 | 独立路径整合，不能称再获一次大C收益 |
| J014 | 合并对齐S=1 tile但重复读取GM权重 | 369通过；收益不稳，低精度[1001,512,1]初测退6–7% | 未采用 |
| J015 | 一次搬一周期C权重，在UB展开 | 399通过；四轮均值[1001,512,1] J013→J015：FP32 45.599→11.303、FP16 43.534→14.132、BF16 43.567→14.135 | 小S=1未触发路径仍可能额外开销，先作专项 |
| J016 | 每核权重驻留、tile取C倍数 | tileLen==C时错误重新截断，原4个低精度C4096失败及扩展4个失败 | 错误版本不进入性能候选，无精度交换 |
| J017 | 上游P105提前读取思路 | 435通过；600k三dtype均值微退且翻转 | 无稳定收益，停止 |
| J018 | 单条件修复J016驻留路径 | 462通过；同批[1201,512,1] FP16 J015 16.669→10.572，BF16 16.384→10.720 | 存在非驻留及小S=1回退，见后表；仅专项 |
| J019/J020 | 普通/合并/驻留独立编译；J020恢复驻留原体 | 各498通过；隔离不能消除大任务后的小输入回退 | 固定顺序证实上下文影响，不能只怪小kernel本体 |
| J021 | 同地址、预热、replay控制实验 | 小目标本身250次预热可消退回退，其他大目标预热不行；ICache证据 | 诊断版，不是算法优化或全局提频证明 |
| J022 | 低精度legacy S=1 ICachePreLoad | 498通过；小输入改善；[2001,512,1] BF16 24.200→25.820 | 大路径没执行该prefetch仍退，可能二进制布局；未作全局替换 |
| J023 | 编译时剪除非当前dtype函数体 | 498通过；代码体明显缩小，但热批次仍有翻转 | [1,1,1] BF16 J022 2.812→2.962，阻止无条件晋升 |
| J024 | split-C按通道块跨batch复用权重 | 582通过；目标三dtype收益明确；[31,32768,1]低精度全退 | 保留其他路径；未证明所有顺序下无影响 |
| J025/J025P | 核数/半tile扫描与选择规则 | 675通过；留出36组9全胜、11全败，平均改善掩盖回退 | 选择规则放弃，不继续按单点补丁 |
| J026 | 缩到少量通道块、足够N范围 | 708通过；保留目标收益并避开已知范围问题 | split-C基线，其他路径按原规则 |
| J027 | 缓存低精度转成FP32的权重、复用缓冲 | 708通过；Vector下降但整算子收益不稳；A/B/A未改FP32也漂移 | 不能从Vector更短推出总时延更短 |
| J028 | 固定tile输入预取双缓冲 | 仅J028IO诊断上板：FP16 7.959→8.512、BF16 7.942→8.528，四轮退；实际算子源码未做精度验证 | 诊断门槛失败后停止；IO identity不是合法算子性能 |
| J029 | 合并队列：每batch API调用16→8 | 723通过；目标三dtype近乎不变且翻转 | Scalar减少未转化稳定总收益；其他14函数字节不变 |
| J030/J031 | 内存/依赖剖析与最小模拟器复现 | Scalar下降而Vector指令队列stall升；模拟定位输入槽晚释放 | 910C核内实板指令timeline不可用；模拟周期不是设备时延 |
| J032 | 低精度Cast完成后提前释放输入槽 | 741精度、1344次压力通过；模拟等待1036→3周期，实板无稳定增益 | FP32别名路径未套用；停止该方向 |
| J033 | 实板第二次Compute分段诊断 | 定位可重复计算成本，非新性能候选 | 插桩时间不能直接充当普通整算子时延 |
| J034 | 低精度UB源地址错开bank group | 744精度、1344压力通过；目标两dtype四轮改善；6/12改动组合四轮全胜 | FP32和单通道控制未改；其他14函数字节不变 |
| J035 | FP32队列地址相位调整 | 750精度、1728压力通过；8个改动组合6全胜，另2翻转 | 低精度沿用J034；冻结基线，非证明任意小输入时序不变 |
| J036/J037 | 冻结、平台/TBE对照、翻转点复核 | 非新算法；直接TBE同批[127,8192,1] FP32 16.592→J035 10.7212、FP16 16.507→11.7411、BF16 16.777→11.9373，四轮同向 | 不是竞赛评分；两个FP32翻转点与顺序敏感仍保留 |
| J038 | 一次初始化持久零向量 | 756精度、2112压力通过；分段下降、17/28改动组合四轮全胜 | 其他14函数字节不变；近期小总量计时未做 |
| J039 | 去两次count Select之间外层同步 | 780精度、5184压力通过；低精度机器码相同；FP32分段反增8–9周期；33组无四轮全胜 | 未晋升；均值小幅下降不能归因删除同步 |
| J040 | repeat Select减少count mask配置 | 780精度、5184压力通过；整算子19/28改动组合全胜，但未改控制也系统性变快；分段基本不改善 | 未晋升，保留证据；不预判必须牺牲小路径 |
| J041 | 小通道/小总量补测准备 | 用户叫停前未启动，无结果 | 本轮仅整理既有证据，不再补测 |

## 小通道到底受了什么影响

**历史上确有回退，但现有证据不支持“当前大通道收益必然要以小通道变慢换取”。** 上述J034/J035的收益来自目标路径地址布局，J038来自目标路径零向量复用；其它函数机器码对照未变。与此同时，J020/J021已经表明相同小kernel也会受此前工作负载、ICache和预热影响，因此不能以“函数未改”宣称任意顺序下性能完全相同。

| 比较与条件 | 小输入 | 基线→候选 μs | 含义 |
|---|---|---:|---|
| J003→J004，两轮普通路径 | [2,3,65] BF16 | 4.760/4.630→5.260/4.910 | 早期确有0.50/0.28回退 |
| J015→J018，同批四轮均值 | [3,5,1] FP32 / FP16 / BF16 | 3.042→3.340 / 3.502→3.995 / 3.195→3.688 | 驻留版引入时存在小路径回退 |
| J013→J020，放在大任务之前 | [3,5,1] 三dtype | 差+0.028/+0.005/−0.025 | 接近持平 |
| 同样小输入放在驻留[1201,512,1]之后 | [3,5,1] FP16 / BF16 | 2.950→3.550 / 2.970→3.592 | 约0.6 μs回退四轮一致，顺序相关 |
| J021对照，先预热该小目标250次 | [3,5,1] FP16 / BF16，J013→J020 | 2.940→2.955 / 2.970→2.970 | 目标预热后回退基本消失；不能混作冷启动成绩 |
| 近期J035/J038/J039/J040 | 真正小N、小C、少元素输入 | **没有近期有效计时** | J041原计划填此缺口，现已暂停 |

近期唯一C很小的性能控制是 `[600000,1,1]`（三dtype），它有60万元素，不能代表小数据量。J038批次J035→J038：FP32 9.1624→9.2499、FP16 8.7284→8.7797、BF16 8.7274→8.8472，四轮方向均不一致。J040批次J038→J040，该未修改控制却分别快0.2590/0.1926/0.2773 μs，说明环境/顺序偏移足以混淆目标收益。

J035/J038/J039的完整控制四轮范围已保存在J040 `SMALL_CHANNEL_HISTORY.md`；范围是四轮统计量范围，不是逐调用范围。当前既不能承诺小输入零回退，也没有执行接受小输入回退的新分派。

## 瓶颈证据、已排除方向和审阅结论

- 权重复用前的目标C8192，J024同批BF16 MTE2 8.815→3.369 μs、MTE3 2.970→1.302；FP16 8.752→3.297、3.322→1.233；FP32 6.838→4.172、2.707→1.946。搬运减少与总延迟下降相符。**这些流水可以重叠，不能相加或当作独占耗时占比。**
- 后续Vector和Scalar活跃时间下降不总能转化成总延迟：J027/J029/J032就是反例。J030的MTE2等待比例59–70%不能证明HBM饱和；L2命中在kernel/application replay间差异很大，不能混用两种模式的结果。
- UB地址布局与实板Compute分段提供了可复现的后续改善证据。模拟器仅用于依赖定位；J032已显示模拟等待消失不等于真实设备更快。
- 低精度双缓冲、继续删外层同步、复杂核数规则均没有形成足够稳健的新增候选。J040整算子正向结果保留，但未改控制同步变快、Compute无对应改善，因果证据不足，不能预先替换J035/J038。
- 已参考的上游包括协作仓P103路径历史、P102撤回、hicann split-C跨batch复用权重思路。J024已按910C实际tile/核数自行适配；并非遗漏后仍重复提出。arch35面向其它型号的实现不能直接当910C可用代码。

现在可供审阅的是：**保留J035作为可回退基线、J038作为增量候选，保留现有普通/小输入路径，不改变取舍。** 若以后选择允许某类小输入回退，应先明确输入范围、冷热/顺序条件和可接受上限；本轮没有替用户作出这项决定，也没有据此再开测试。

## 精度与来源边界

后期精度/压力包含三dtype、对齐/尾块/路由边界、有限正负随机值和零；不是所有IEEE NaN/Inf与有符号零语义的穷尽证明。J016失败版本及未上板验证的实际J028不列作可用候选。提交1109131后来取得的源码并非J001；尚未复现其平台WA根因，不能将平台WA归因当前版本。

本审阅没有重新运行测试。汇总CSV/JSON保留每个同批对照的形状、dtype、四轮统计、采样窗口和本地分析源文件SHA256；原始数据、源码、档案校验与详细命令以各实验报告为准。

### 逐版报告索引

[J002](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J002/PROGRESS-910C-20261104.md) · [J003](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J003/README.md) · [J004](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J004/RESULTS.md) · [J005](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J005/RESULTS.md) · [J006](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J006/RESULTS.md) · [J007](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J007/RESULTS.md) · [J008](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J008/RESULTS.md) · [J009](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J009/RESULTS.md) · [J010](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J010/RESULTS.md) · [J011](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J011/RESULTS.md) · [J012](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J012/RESULTS.md) · [J013](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J013/RESULTS.md)

[J014](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J014/RESULTS.md) · [J015](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J015/RESULTS.md) · [J016](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J016/RESULTS.md) · [J017](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J017/RESULTS.md) · [J018](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J018/RESULTS.md) · [J019](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J019/RESULTS.md) · [J020](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J020/RESULTS.md) · [J021](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J021/RESULTS.md) · [J022](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J022/RESULTS.md) · [J023](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J023/RESULTS.md)

[J024](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J024/RESULTS.md) · [J025](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J025/RESULTS.md) · [J026](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J026/RESULTS.md) · [J027](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J027/RESULTS.md) · [J028](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J028/RESULTS.md) · [J029](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J029/RESULTS.md) · [J030](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J030/RESULTS.md) · [J031](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J031/RESULTS.md) · [J032](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J032/RESULTS.md)

[J033](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J033/RESULTS.md) · [J034](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J034/RESULTS.md) · [J035](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J035/RESULTS.md) · [J036](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J036/RESULTS.md) · [J037](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J037/RESULTS.md) · [J038](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J038/RESULTS.md) · [J039](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J039/RESULTS.md) · [J040](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J040/RESULTS.md)

小通道已有数据：[SMALL_CHANNEL_HISTORY.md](E:/suanzi/cann-ascendc-ops-collab/people/JadeYara/PReluGradUpdate_2329/experiments/J040/SMALL_CHANNEL_HISTORY.md)。

J040完整归档：136263117字节，SHA256 `fb8e31910c85fdbd95b7f24b2526d68bf0a8ac2271416246abca352c7d595e47`；此前CRC与6810个二进制路径、3973个唯一引用、14个依赖核验通过。

基线源码ZIP SHA256：J035 `212183985a1795c76ca2fddf298d53ca4229321cdcdaa87517882b8ae17bbd35`；J038 `28bcdbec7382290e395c9e78bb48667e4cd8b11b394b192c4969f0706c92a4a6`；J040 `c8774543b974e8828e1f2206b3be8554e04707d9ab33824f3defd8e6dc05f03a`。这些是此前已校验记录，本轮只读取证。
