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
