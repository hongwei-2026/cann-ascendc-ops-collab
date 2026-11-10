# J047：复核 J027 转换缓存与 J035 UB 布局的交互

先审旧证据：J027 基于 J026，仅 split-C FP16/BF16 将权重首 tile Cast 到既有 negFloatBuf 常驻；Compare 后复用 featuresFloat 写 weight*grad，再 Select dx。708精度通过；普通63输入无稳定收益，固定A/B/A目标低精度约-0.85/-0.766us，但未改FP32也-0.326/-0.341us。旧kernel/application replay仅J026/J027IO，没有J027，不能据此声称缓存候选已经做过冷/热对照。

实质遗漏条件：逐字比对46文件，J026→J035唯一差异是splitc.h的UB分配布局。J034实板Compute bank-group冲突和时间下降，改变了当前基线；旧J027从未与新布局组合、也未做对称ABBA与无profiler对照。一次有界2x2：旧布局J026/J027，新布局J035/J047；候选只把已经验证的J027 Compute改动原样移植到J035。保持Host/tiling/核数/队列分配/输入遍历/权重GM驻留，FP32不改。新布局保持Mul/Select双输入0/256相位，实际收益待实测，不能把相位推导当硬件计数器。

780三dtype精度和57案例5184逐调用压力；性能仅15输入=三个大形状[127,8192,1]/[81,8256,1]/[91,10368,1]与两个小控制[127,32,1]/[4,6,8]×3dtype。每布局内部四轮ABBA/BAAB，交替布局顺序、后两轮反转case顺序。用9输入无profiler enqueue复核，保持协议分开。只实施这一候选，不继续扫描；若无可复现收益则停止权重转换缓存方向。没有新的MemoryDetail或仿真作业。用户负责官方提交，本轮不操作平台/不push。
