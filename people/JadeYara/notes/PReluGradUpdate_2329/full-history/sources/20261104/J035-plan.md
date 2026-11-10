# J035：FP32 split-C 队列地址布局

本轮独立验证 J034 之后的一个候选，保留 J026/J034。设备、工具、原始证据沿用 [J034 报告](../J034/RESULTS.md)。

J033/J034 的实板分段已测得：FP32 第二批 Compute 为 0.740606 μs，bank group 冲突比例26.0229%。FP32 的输入输出直接别名队列，因此不能套用低精度临时 float 缓冲的配置。

基于 J034 创建 J035，只调整 FP32 split-C 的分配分支。令 A 为对齐64元素的 tile 长度，F=AlignUp(4A,512)，M=AlignUp(A,512)。五个队列和两个 float 临时缓冲以 F 为基本长度；grads、weights 队列各额外留256B；mask 分配 M+256B。各源地址模512的相位如下：

| tensor | 相位（字节） |
|---|---:|
| grads | 0 |
| features | 256 |
| weights | 256 |
| dx / zero | 0 |
| update | 0 |
| neg | 256 |
| fg | 256 |

grads/weights、features/grads、features/zero、zero/fg、grads/neg 五组二元源对因此错开256B。该规则适用于任意对齐后的 tile；没有形状名单，也不扫描 padding 参数。对物理 bank 的读写冲突不预设已经消除。

仅改变地址分配，不改变数学、Compare/Select、Cast、同步、队列深度、tile、分核和路由。低精度继承 J034，编译后校验函数字节。最大 FP32 tile=5184 时，7F+M+768=153344B，比原分配多3008B，低于保留8KiB后的188416B可用UB。

J035C 是增加相同第二批 Compute 采样窗口的诊断副本；真实性能候选为不插桩的 J035。首先750组精度（744已有+两个新形状×三dtype），27组合×64次逐次输出压力，再两轮反向顺序分段对照。新形状 `[91,10368,1]` 覆盖FP32两块最大tile，`[107,15424,1]` 覆盖三块最大tile与尾块。

若分段冲突和Compute没有改善，则停止；若改善，再进行J034/J035四轮ABBA/BAAB完整Task对照，27个dtype/shape组合，每组250次，统计101–250次。低精度与非split-C路径作为未修改控制。此轮仅验证一个候选。
