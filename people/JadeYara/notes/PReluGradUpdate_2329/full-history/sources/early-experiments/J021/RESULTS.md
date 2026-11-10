# J021：大输入后小输入回退的控制实验

本轮证据支持**目标代码的指令缓存状态/预热不足**是 J020 小输入顺序回退的主要解释：复用 GM 地址不能消除；目标 tiny 预热有效，无关大算子预热无效；慢样本 ICache miss 与 AIV cycles 同步升高。后续 J022 针对该证据做指令预取实验，结果见相邻 J022/RESULTS.md。这里没有将管线活跃占比当成独占耗时。

## 环境与工具口径

Ascend910_9362，原生 ascend910_93 / dav-c220-vec，CANN 9.0.0，driver 25.5.5。未安装或升级系统工具，未修改网络、安全设置、已有目录权限、频率。普通 `msprof --aic-metrics=PipeUtilization --task-time=on` 和 `msprof op` 分开统计。安装工具支持两种 replay，但 `msprof --version` / `msprof op --version` 均不支持；现有 profiler 可执行文件 SHA256 为 `3bc7142b4d84e7be2f433700cf232dfb911f28fba865afb567e4701342bf91da`，CANN header NUM=90000000。

## 普通 profiler：地址、顺序、预热控制

J013/J020，fresh 每组分配、reuse 预分配并反复使用相同 GM 地址；3 dtype，4 个轮次反转版本/分配模式顺序。tiny=[3,5,1]，resident 标记输入=[1201,512,1]，merged=[2001,512,1]，scalar=[600000,1,1]。resident 标记只在 FP16/BF16 对应驻留，FP32 为 merged。每个 tiny 250 次；按 1–5、6–25、26–100、101–250 次分窗，不将冷热混合。前驱25/250次、10ms间隔和同址H2D重写均纳入。16次进程运行，624组输出校验，112800条 kernel profile 记录，全通过。8份 reuse 日志核实 tiny 地址一致。

fresh/reuse 结果接近；初次 tiny 在6–25次窗口仍可明显偏慢，101–250次稳定后 ICache miss 近零。目标 tiny 预热后，大算子25/250次、10ms间隔和H2D重写均未恢复原来的持续0.6us回退。

针对性控制：FP16/BF16，2轮反向版本，cold=tiny25→large25→tiny250；tinywarm=首个tiny250；scalarwarm=先无关600k250再cold序列。24次进程、80组输出、11000条profile记录，全通过。表中为后续 tiny 第6–25次窗口中位数在两轮的平均，单位μs；ICache miss 是工具原始比率。

| dtype | 前置条件 | J013 Task | J020 Task | J013 ICache miss | J020 ICache miss |
|---|---|---:|---:|---:|---:|
| fp16 | cold | 2.980 | 3.445 | 0.0000 | 0.0377 |
| fp16 | tinywarm | 2.940 | 2.955 | 0.0000 | 0.0020 |
| fp16 | scalarwarm | 2.980 | 3.650 | 0.0000 | 0.0423 |
| bf16 | cold | 2.995 | 3.565 | 0.0000 | 0.0390 |
| bf16 | tinywarm | 2.970 | 2.970 | 0.0000 | 0.0000 |
| bf16 | scalarwarm | 2.970 | 3.490 | 0.0000 | 0.0367 |

旧固定位置数据中 BF16 的 J013→J020：Task 2.970→3.592us，AIV 2.434→3.066us，cycles 4016→5058，ICache miss 0→0.038。cycles/AIV time 来自工具聚合换算，不能单独用来证明实际时钟。只读 npu-smi 空闲快照 curFreq=800MHz；它不代表短 kernel 执行频率。

## msprof op：replay、工具预热、间隔

固定 BF16 tiny25→large25→tiny1，skip50/count1，J013/J020×kernel/application×warm-up5/250×gap0/10ms×2反向轮，共32次profile运行；每次最终3组输出均正确。先用独立 pilot 验证参数和CSV（pilot 不进入下表）。application 每次工具采集实际重跑应用8遍；kernel 直接重放所选kernel。两者结果不能混平均；工具warm-up也不等同于程序里显式重复目标算子。

[官方固定版本指南](https://github.com/Ascend/msopprof/blob/16beabbbb76f12136cc940866990b7f3e0289602/docs/en/user_guide/msopprof_user_guide.md) 指出 kernel 模式清L2、application 保留之前的L2状态，并提醒短算子预热可能不足。下面为本机实测，非引用指南数值。

| replay | warm-up | gap μs | 版本 | Task μs（两轮均值） | ICache miss | L2 read hit % | Current Freq MHz |
|---|---:|---:|---|---:|---:|---:|---:|
| kernel | 5 | 0 | J013 | 3.160 | 0.0000 | 57.14 | 1650 |
| kernel | 5 | 0 | J020 | 3.160 | 0.0000 | 68.42 | 1650 |
| kernel | 5 | 10000 | J013 | 3.230 | 0.0000 | 57.14 | 1650 |
| kernel | 5 | 10000 | J020 | 3.230 | 0.0000 | 68.42 | 1650 |
| kernel | 250 | 0 | J013 | 3.170 | 0.0000 | 57.14 | 1650 |
| kernel | 250 | 0 | J020 | 3.230 | 0.0000 | 57.14 | 1650 |
| kernel | 250 | 10000 | J013 | 3.110 | 0.0000 | 57.14 | 1650 |
| kernel | 250 | 10000 | J020 | 3.230 | 0.0000 | 57.14 | 1650 |
| application | 5 | 0 | J013 | 3.110 | 0.0000 | 92.86 | 1650 |
| application | 5 | 0 | J020 | 4.070 | 0.0794 | 98.25 | 1650 |
| application | 5 | 10000 | J013 | 2.980 | 0.0000 | 92.86 | 1650 |
| application | 5 | 10000 | J020 | 4.140 | 0.0789 | 98.25 | 1650 |
| application | 250 | 0 | J013 | 3.090 | 0.0000 | 92.86 | 1650 |
| application | 250 | 0 | J020 | 4.220 | 0.0794 | 98.25 | 1650 |
| application | 250 | 10000 | J013 | 3.150 | 0.0000 | 92.86 | 1650 |
| application | 250 | 10000 | J020 | 4.090 | 0.0794 | 98.25 | 1650 |

application 的 J020 更慢且 ICache miss 较高，即使L2命中率更高；kernel 的 ICache miss 为0。两模式已报告的当前频率均1650MHz，目标预热与无关预热对照也不支持“纯整体低频”解释。这仍不是高频率采样的电源状态追踪，不能穷尽所有 DVFS 机制。replay 同时改变应用生命周期和缓存状态，不能用此比较独立量化L2清除代价。

初次 pilot 因本地生成的CRLF shell未执行，已保存错误后改用独立LF脚本；下一次被工具拒绝输出目录group-writable，未采样。通过新建仅当前用户可写的专用目录完成预检，未 chmod 任何已有路径。所有错误日志保留。`--version` 不支持、一次 npu-smi 查询参数错误同样保留。

## 上游对照与结论边界

另见 J021-upstream-comparison.md，固定 hicann/ops-nn b5f577d6、CANN OSL2许可。其 experimental PReLU ascend910b split-C 遍历同通道块的batch以复用权重；J018/J020是连续平坦数据分块下的完整C驻留，并未实现split-C。该思路保留为大通道独立候选，没有拿950/arch35生产实现宣称910C兼容，也没有移植代码。

后续 J022-hot 控制中，FP16 J020 的大输入后第6–25次仍有3.240us、ICache miss 0.019的过渡回退，而101–250次归零并稳定2.950us；因此目标预热有效是本次条件下的证据，不是对任意后继序列的缓存驻留保证。

本轮诊断的是指定本地输入、顺序与二进制组合；不是未知榜单输入的计时。4–5us目标与1109131提交来源不明的边界均保持不变。

## 完整证据

`E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J021_J022_ICACHE_910C_20261104\J021-J022-910C-evidence.zip`

字节数 155428624，SHA256 `bb29ad1c55ec9fecc69c2c0e18c3fcaedd6cc06b342876d30b0fca8688686e1d`。本地ZIP CRC、7021条binary路径/哈希、2788个引用blob、两份前置archive SHA、输入golden哈希及源码绑定全部核验；JSON/CSV/日志/脚本展开，原始二进制profile与tensor按binary-index去重保存在ZIP或明确依赖的前置ZIP。RESTORE.txt/DEPENDENCIES.json给出完整恢复方式，不能只复制此增量ZIP后丢弃依赖。旧候选原文件均保留。没有竞赛提交或git push。
