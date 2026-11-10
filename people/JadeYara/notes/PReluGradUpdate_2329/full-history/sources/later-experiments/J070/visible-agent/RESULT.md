# J070 实板结案：泛化正确性通过，尚无整体性能突破

2026-11-08，JadeYara，可见代理独占接续。原 910C、CANN 9.3、ascend910_93。当前保留 J070 为独立泛化正确性基线，生产 J035 未替换，未官方提交、commit 或 push。

## 已修复

1. 补齐标准/B 的 InferDataType 注册。
2. 为模板 key 显式选择 Old80 / Gather88 / General760 元数据；标准9个、B9个编译描述的 opParaSize 均为768。
3. NCHW 外部 ND 权重被 GE 内部标为 NCHW、存储仍为 [7]；Host 现在兼容该连续存储标签，保留元素数和广播校验。
4. NC1HWC0 实际格式值为83886083，含子格式位；使用 SDK GetPrimaryFormat 比较主格式，保留实际物理维度及存储。

attempt4/5 仅修改 Host/推导；核代码、数学链、key 和生产版本不变。标准/B 189/189 编译函数逐字节一致，A/C54/54一致；B/A旧函数39/54一致，所有差异保留并完成实测。

## 精度证据

- 63个格式/广播/空维/特殊值提议中，内置TBE接受61个；标准和B均通过，双输出逻辑差异0。
- TBE拒绝 probe_scalar_grads、probe_rank1_features_scalar_weight，保留为合同拒绝，不计作候选通过。
- FP16、BF16各覆盖65536种feature位型的广播输入：两种算子均通过，逻辑差异0。
- 原1063组回归：ABC3189窗口/4584调用，标准1063窗口/1528调用，全部通过。
- ND30例来自attempt3；NCHW9例来自attempt4；NC1/其余/全位型和全部旧回归在最终attempt5完成。后两次修复只改变此前被拒的格式判断，未改ND计算。

## 性能证据

旧路径相对同轮J069：127输入、1066800调用、762轮。基线A/A/C控制762/762稳定，全部四路同速控制757/762。5个例外全部为FP32 [4097,1,1]、wlen1，候选略快而越过2%的“同速”门槛；未丢弃、未补跑。各输入中位相对变化的中位值为快0.0081%，基本持平；最慢输入BF16 [4097,1,1]约慢0.56%（约0.015us）。

新增通用路径对内置TBE：12输入、100800调用、72轮基线控制全部稳定。六种顺序交错，同一进程/五个缓冲区/stream，每窗350次，弃前200、留150；每窗末检查输出。下表为设备Task Duration的六轮中位数汇总，增益正值表示候选更快；不是官方评测成绩。

| 用例 | TBE μs | J070 μs | 中位相对增益 |
|---|---:|---:|---:|
| nd_complementary_fp32 | 7.5326 | 9.2305 | -22.54% |
| nd_complementary_fp16 | 11.4750 | 10.7400 | +6.32% |
| nd_complementary_bf16 | 11.3001 | 10.4400 | +7.70% |
| nd_window_boundary_fp32 | 14.8504 | 27.8407 | -87.35% |
| nd_window_boundary_fp16 | 15.3201 | 30.0053 | -95.95% |
| nd_window_boundary_bf16 | 15.3003 | 29.9005 | -95.49% |
| nchw_channel_fp32 | 4.3603 | 8.3400 | -91.27% |
| nchw_channel_fp16 | 4.7527 | 8.8400 | -86.00% |
| nchw_channel_bf16 | 4.7576 | 8.8400 | -85.81% |
| nc1_c19_broadcast_g_fp32 | 3.9075 | 10.5553 | -170.08% |
| nc1_c19_broadcast_g_fp16 | 4.4150 | 9.9352 | -125.14% |
| nc1_c19_broadcast_g_bf16 | 4.4200 | 8.9153 | -101.70% |

仅互补广播FP16/BF16比TBE快约6.3%/7.7%；其余慢22.5%至170.1%。当前通用路径未达到普遍TBE95%性能要求，不能宣称整体提速或替代生产基线。

## 交付与边界

- 最终标准源码：[candidate-standard-source-attempt5.zip](attempt5/candidate-standard-source-attempt5.zip)，51文件，SHA256 `9b8320429169af39ffd1529a35a6418ad9a21130855222793d705c7fa79b200e`。
- 证据包：[device-evidence.zip](device-evidence.zip)，60544364B，1432文件，SHA256 `ab5ce45569d476fc69664dd34d5d456b6c3b85392f87c2bc71c53f5f0051d0f4`。整包SHA、逐文件清单SHA/大小及ZIP读取CRC均通过；51个标准源文件与云端实测源码逐字节一致。
- [结构化结果](result-summary.json)、[控制例外](legacy-control-exceptions.json)、[校验记录](archive-meta.json)。完整成功/失败文本日志、输入/输出、全部CSV、Host/Kernel二进制在证据包内；原始profiler数据库和可重建的可视化时间线JSON继续保留云端完整253319592B包，详见包内ARCHIVE_SCOPE.json。下载未完成的.part不是有效证据。
- 云端目录 `/mnt/workspace/2329-JadeYara-20261104-1812/J070/attempt3-visible` 保留attempt3构建及后续Host修复，原始J070构建、failure-r1、初次编译快照仍在。
- 生产J035的46文件与接手前哈希完全一致。未进行官方评测；未知rank图编译及未覆盖描述仍不作通过承诺。
