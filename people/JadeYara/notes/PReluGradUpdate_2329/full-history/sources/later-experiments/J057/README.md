# J057：FP32大通道整行归属，验证后不采纳

2026-11-06，JadeYara。已结束；不替换J035或J056，没有官方提交或公开推送。用户最新要求先给方案、由用户决策，后续实验暂停。

## 结果

- [127,8192,1] FP32：A/J035 10.8475 μs；B/整行 11.660 μs；C/J054两行控制 11.265 μs。B比A慢7.49%，六轮均慢。
- 9个实际启用输入中，B没有一个输入在任何一轮胜过A。范围内的退化约0.23%–11.50%，不从中挑点推广。
- 精度：780个已有输入加9个新增输入，A/B/C共2772次调用，2367个窗口，两个输出逐位比较通过，仅允许正负零等价。原始NaN位型诊断与修正边界见下文，不宣称IEEE穷尽验证。
- 性能：22输入、528窗口、184800条内核记录；六种平衡顺序，同一进程、五个张量与同一流。每窗口350次，舍弃前200次，取后150次Task Duration中位数。六轮是进程内平衡顺序，不是六次重启。
- 132组控制全部通过（≤0.1 μs且≤2%）；A/A最大0.040 μs。不是明显控制漂移引起的失败，不重复测量刷门槛。

完整逐形状与逐轮结果见evidence/overview.json和evidence/perf/summary.json。

## 本轮只改变什么

A使用冻结J035测试别名。C复用已否定的J054两行版作为机制控制，没有重新调优它。B沿用J054的四长向量生命周期、FP32 Compare/Select及同步顺序，但将一个完整通道行归属给一个核。

B仅在原有FP32 split-C编译路径、原tile≤4096、C≤8192且C为64的倍数时启用；Host、分派数据与其他dtype不变。每核按原coreNum划分batch区间，整行连续读取，权重常驻。它与J054两行跨C切块的区别是行归属、连续DMA和单次整行Mul，不是重跑同一变体。

C=8192时最大显式UB分配165632B，低于本机可用上限；没有依靠缩小原Host tile或减少核数容纳缓冲。小输入、非整C与S>1维持回退路径。最终未制作可提交候选。

## 不能误读的局部指标

目标A1/A2的Scalar活跃时间中位数约4.535 μs，B为2.487 μs；Vector约2.306→1.958 μs。MTE2约4.198→4.162 μs，MTE3约1.989→1.823 μs；但完整Task变慢。

这些活动时间可以重叠，不能相加，也不能据最大一项认定唯一瓶颈。现有PipeUtilization不是完整指令/事件关键路径。结论只是：减少循环与局部活动不足以保证总耗时改善。

## 验证边界与失败原文

首次新增极值输入包含0×Inf。严格逐位比较先在A/J035失败，element8199：硬件0x7fffffff、NumPy 0x7fc00000，均为NaN但payload不同。B尚未执行该输入，不能归因为新核错误。

保留inputs/new_8、precision/runner.log、tests.exit=2与nan-diagnostic-env.log。比较器只增加错误bit值输出，判定逻辑不变。替代用例inputs/new_8_no_nan使用非零有限grads/weights，仍覆盖±0、±Inf、最大值、最小正规值及下溢结果，避免无效运算产生NaN。原有及前8项通过证据复用，最后一项A/B/C各16次单独完成。precision/prefix-validated.json与precision-resume/runner.log分别支撑最终汇总。原NaN输入没有被标为逐位通过。

诊断过程中还出现缺少driver库路径导致链接失败、脚本换行转义错误和缺少完整CANN环境导致ACL 361001，均保留日志；恢复原构建/运行环境后定位成功。未改系统安装或网络设置。

A的45个函数体与J053冻结A保持一致。三别名135个函数体中125个相同；B/C的4个FP32回退函数长度相同但字节变化，不能宣称全部回退机器码不变。其他dtype保持对应机器码一致。详见evidence/function-bodies.json。

## 复现与证据

- code/、control/、plan.py、build.sh、tests.sh是首次实验入口。
- correct-fixture-remote.py和evidence/resume.sh记录精确的测试数据修正及剩余检查，不重复已通过用例。
- evidence/保存所有成功/失败文本日志、计划、汇总及函数字节对照。J057-text-evidence.zip：287868字节、86个文件；SHA256 0d8f83ed572d901332847f7d6f056a5624d3547d4a4b73ccb38146ee70dd03cf，下载后校验一致。
- 原始二进制输入与未压缩profile保留在远端个人J057目录的inputs/和perf/profile/，本地未下载全部profile。
- 最终resume.exit=0；23:57 CST复查无J057/msprof/device_probe作业运行。J035主workdir无diff。

## 参考来源与已有止损

已读个人J035、J046–J056记录，共享P102/P104回退和P105准备权重前移记录。没有重新启用全面双缓冲、每tile重读权重或NaN不等价的掩码乘法。

昇腾社区官方[搬运优化文章](https://www.hiascend.com/zh/developer/techArticles/20240906-1)建议较大连续数据块及利用stride参数减少小块循环，但明确收益受芯片/频率影响。它是本轮假设的依据，不是胜利证据。

当前[2201架构参考](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html)说明UB预算、对齐与事件依赖。本站为master预览；实际构建仍以已安装CANN9.3头文件及本机检查为准。

官方J056提交1191764由用户确认，Pass，五点7.21/7.16/7.19/5.58/5.84 μs。官方shape/dtype未知，不与本轮本地输入绑定。

## 决策

详见DEC-J057.md。本路线关闭，不合入、不试交、不追加调参。后续仅提供方案，等待用户选择。
