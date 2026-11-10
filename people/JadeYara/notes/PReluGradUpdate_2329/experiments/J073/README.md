# J073：General Host 核数分配

2026-11-08，原910C / CANN9.3 / ascend910_93。未通过预注册门槛，不晋升，保留J071。

9个受影响小输入中，9个输入级中位延迟改善≥5%，改善范围24.58%–48.94%。3个窗口边界输入保持原全核规则。General基线控制72/72、TBE重复控制72/72、旧路径控制761/762。全量值见下表，异常不删除、不选择性重测。

## 附件与断点

- 文本附件 `libfile_ffaa2636042c81918ea594d6aca7f9b0` 为DeepSeek调研建议，不是官方执行日志。读取全文；P2-a建议只改General核数，固定规则、不扫阈值，并保持真TBE计算链。
- 截图 `libfile_e9888d98743c819198f72c2aade67f2f` 经父线程独立像素核读：五点Pass、误差0.00%，7.25/7.33/7.44/5.67/5.73μs。截图无版本/提交ID，不能独立绑定为J071官方成绩。相对此前J069记录分别少0.30/0.10/0.12/0.10/0.10μs，仅作数值对比，不推断shape或因果。
- 初次本地图片读取受Library工具在Windows缺少os.setxattr阻碍，未生成本地图片；没有绕过工具下载。最终数值使用父线程独立核读，版本归属仍未确认。
- 核对已有J073源码及云端build/campaign后接续，未重复启动编译或实验。J072不继承。

## 唯一改动与选择理由

只改 `op_host/p_relu_grad_update_tiling.cpp` 的General分支：`coreLimit=min(hardware,max(4,ceil(total/2048)))`，据此重新计算perCore、对齐blockSpan和used；空张量逻辑不变。没有只截小used而漏掉元素。140112组静态整数模型检查覆盖不同硬件核数、2/4字节dtype及边界规模，均满足连续覆盖、对齐和核数上限。

内核源码逐文件与J071一致；旧路径在General分支之前返回，未改其路由、tile或核数。真实TBE数学、量化回读、dx*0依赖、Inf/NaN行为均沿用J071。

依据是J070/J072已测General小输入相对TBE的延迟缺口，以及附件的P2-a假设。公开每核ramp时间未在本题独立测量，不当作结论。J025/J026是其它旧路径的减核试验；本轮只改变General的固定结构规则。

J072 BF16回退根因仍未决。已核对其Host仅General预算一行变更，旧路径在之前返回；其General性能和旧路径性能使用独立进程，旧路径runner内部不执行General。此事实不排除跨进程运行态或包级影响。没有用dummy/padding或禁掉未执行路径来直接宣称因果。本轮通过完全沿用J071内核并在计时前验证所有函数体，隔离J072内核改动；不声称解决了J072归因。

## 构建、精度及机器码

- 标准及ABC构建完成；69个公开ACL提议中67个被真实TBE接受，标准与B全部通过，逻辑差异0。两项拒绝保留：scalar grads及rank1 features/scalar weight。
- 复用此前真实TBE oracle，验证输入、元数据和golden哈希后重新执行候选；覆盖3 dtype、ND/NCHW/NC1、rank8、广播、空维、尾块、特殊值，以及FP16/BF16全部65536特征位型。
- 1067组旧路径精度：ABC4776次、标准1592次调用全部通过。
- 标准/B、B/A、C/A的567项函数体比较全部一致；标准候选与已测J071的189函数体也全部一致。此门禁在性能计时前执行，失败则停止。
- 标准包的9个内核对象及对应9个JSON元数据，与J071逐字节一致（18/18），见[kernel-package-vs-J071.json](evidence/kernel-package-vs-J071.json)。本轮实际内核包也保持一致，Host库是有意改变的唯一执行代码。
- 635个旧性能输入/golden文件的试前、试后哈希一致。

## 完整 Task Duration

12输入×6轮，顺序TABCAT/TCAABT/TBACAT/TACBAT/TBAACT/TCABAT；A1/A2/C是J071，B是J073，T1/T2是本轮实际执行的内置TBE。每窗口350调用，舍前200取后150中位数，共151200调用。同进程、流、固定张量地址；每窗口末核对双输出。表中正改善表示候选更快，不把内部Vector时间当作整体收益。

时延列为六轮各自窗口中位数再取中位（J071每轮先平均A1/A2，TBE先平均T1/T2）；改善率为六轮配对比值的中位数。因此改善率不必精确等于表内两列中位时延直接相除的结果。

| 输入 | J071 μs | J073 μs | 内置TBE μs | 相对J071改善 | 相对TBE改善 |
|---|---:|---:|---:|---:|---:|
| nd_complementary_fp32 | 10.8400 | 6.8100 | 8.9125 | +37.18% | +23.57% |
| nd_complementary_fp16 | 9.0025 | 6.7900 | 11.2500 | +24.58% | +39.68% |
| nd_complementary_bf16 | 9.1600 | 6.8000 | 11.4250 | +25.76% | +40.51% |
| nd_window_boundary_fp32 | 27.8200 | 27.8100 | 14.8150 | +0.02% | -87.88% |
| nd_window_boundary_fp16 | 29.9375 | 29.9500 | 15.1200 | -0.04% | -97.98% |
| nd_window_boundary_bf16 | 30.8825 | 30.9300 | 16.3000 | +0.00% | -89.70% |
| nchw_channel_fp32 | 8.3400 | 4.5450 | 4.3350 | +45.47% | -4.84% |
| nchw_channel_fp16 | 9.4200 | 4.8100 | 4.8825 | +48.94% | +1.38% |
| nchw_channel_bf16 | 8.6200 | 4.7600 | 4.5900 | +44.78% | -3.48% |
| nc1_c19_broadcast_g_fp32 | 8.8300 | 5.0800 | 3.8575 | +42.47% | -31.78% |
| nc1_c19_broadcast_g_fp16 | 8.3775 | 5.3100 | 4.3900 | +36.62% | -20.89% |
| nc1_c19_broadcast_g_bf16 | 8.9300 | 5.3500 | 4.4125 | +40.11% | -21.32% |

General获胜61/72轮，严格获益54/72轮（>0.1μs且>2%）。基线控制门槛≤0.1μs且≤2%；全部异常见[result-summary.json](evidence/result-summary.json)。

本轮仅4/12个输入的中位延迟快于TBE。窗口边界仍明显慢于TBE；局部改善不能写成全场景性能达标，也不能映射到官方隐藏点。

旧路径127输入，共1066800调用/762轮，严格控制761/762；[逐例表](evidence/perf/case-summary.json)及summary中的legacy.exceptions/worst完整保留。

### 旧路径控制异常明细

- 输入[64, 96, 33] / fp16，第5轮：A1=12.610000、A2=12.620000、B=12.620000、C=12.480000μs；A/A稳定=True。

本轮唯一异常为C副本相对A/B快约0.14μs，超过预注册0.1μs绝对门槛；B本身与A接近。这不是候选B出现>2%输入级退化的证据，也不能作为删除该轮、改门槛或选择性重测的理由。异常原因未定位，故严格门槛仍未通过。9个受影响小输入54/54轮严格获益的事实独立保留。

## 门槛与交付

事前门槛：至少6个受影响小输入中位改善≥5%；窗口边界无>2%退化；旧路径无>2%退化；精度与原严格控制均通过。只测这一个候选，不扫阈值，不追加新方向。

**未通过预注册门槛，不晋升，保留J071。** 本轮未官方提交、commit或push。J071、J072源码及生产J035的46文件均与试前哈希相同。

- 标准源码包（本地归档：`J073-standard-code.zip`，本次不上传 ZIP），code/顶层，SHA256 `87bf0749f5fd25ad3567b3aab608d425111bc3bbf3c0ac92a3f730a2cff736a7`。
- 证据包（本地归档：`device-evidence.zip`，本次不上传 ZIP），61339477B/1421文件，SHA256 `487cadf278d553dd983a7b47044790ca2e595a566d0e1390b1dd3b25f0d8a306`；整包和全部清单文件SHA/大小/ZIP读取CRC验证通过，标准/ABC源码匹配云端实测源码。
- [最小diff](host.diff)、[源码检查](source-check.json)、[验证记录](evidence-meta.json)。
- 云目录 `/mnt/workspace/2329-JadeYara-20261104-1812/J073`。原始profiler数据库留云端，全部CSV、日志、公开ACL输入输出、实际编译包和脚本已归档；继承旧输入按manifest路径/哈希保留，范围见包内ARCHIVE_SCOPE.json。

归档工具包上传首次被自动审批拒绝（未执行），理由为云端归属及本题传输授权未确认。读取用户2026-11-07明确授权原始消息后，仅重试相同上传一次并成功；未换目标或绕过审批。


## J074 follow-up diagnostic

2026-11-08: Fixed forward/all-A/reverse diagnostic completed:302400 calls,864/864 correctness windows,216/216 controls; mixed-order B-A/C-A median differences0. Original round5 C drop continued into all aliases in round6. No persistent candidate regression or fixed C advantage reproduced. J073 retained as a user-trial candidate; original761/762 and original decision above remain unchanged. Physical cause and official performance remain unverified. See [J074 report](../J074/README.md).


> 2026-11-10 同步注：本页为原实验记录。仅公开选定文本证据，不表示所有相对路径产物均已上传；代码状态和下一步以[当前进度](../../progress-20261110.md)为准。
