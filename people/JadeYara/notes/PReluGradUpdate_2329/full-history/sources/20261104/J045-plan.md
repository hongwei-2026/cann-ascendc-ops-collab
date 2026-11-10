# J045：固定 tile/核数的连续通道 tile 遍历

JadeYara / PReluGradUpdate_2329。响应用户架构复盘与本人提交安排；本轮只做一个可证伪架构对照，不操作在线工程、不提交、不推送。

历史核验：J023的S=1路径已经向量DataCopy权重，不是P105逐元素展开；J024与它有同机对照且大目标明显改善。但两者同时改变tile切分/任务分布/UB布局，尚无当前J035固定tile/核数/Compute的严格flat对照。因此不能说flat从未测试，也不能直接把旧J023的17us作为此候选结论。

基于冻结J035，唯一文件op_kernel/p_relu_grad_update_splitc.h。保留Host分支条件、blockDim、tileLen、UB分配顺序/大小、CopyIn/Compute/CopyOut及同步数学；只改任务遍历与权重驻留策略：原每核固定C块跨batch，改为线性通道tile连续区间，每tile向量重读权重。使用既有nonresident队列释放逻辑，无逐元素权重展开、无Gather、无新设备API。

ctiles=ceil(C/tileLen)，totalTiles=N*ctiles，每核flat区间[floor(totalTiles*core/cores),floor(totalTiles*(core+1)/cores))；每tile在当前C块内，尾块截断；当前输出段后紧接下一段，跨batch时权重offset归零。保持tile数量和核数；有C尾块时每核字节负载分布可能改变，不将其混作仅缓存效应。

对[127,8192,1]仍40核、tile4096、每核6/7tile。权重读取由20*C增至127*C，逻辑GM总元素由4*N*C+20*C变为5*N*C，增加20.27%；如果连续遍历仍更快，才支持跨batch调度/地址组织的代价超过权重节省。若变慢，则保留split-C，不靠未知官方shape推翻本轮结果。不能从GM逻辑字节直接推算HBM带宽。

先索引覆盖检查及三dtype780精度、57输入5184次逐次双输出压力；通过后J035/J045同机交替，普通msprof Task及少量无profiler enqueue事件区间分开记录。J044已知跨进程约0.4–0.6us漂移；细小均值变化不宣称收益。保留小输入和不走split-C的大输入控制，不用有Block Num异常的图PMU。

官方1148578由用户报告Pass，5点7.15/7.29/7.11/5.88/6.06us；其实际全量源码未取得，不能绑定J035。若完整文件指纹等于J035，候选只需替换上述一份header；若不等，先取得1148578完整回退包，不能把单header混入未知Host/tiling ABI。交给用户的候选会含完整源包、单文件diff、SHA清单及条件化替换/回退步骤，平台由用户操作。
