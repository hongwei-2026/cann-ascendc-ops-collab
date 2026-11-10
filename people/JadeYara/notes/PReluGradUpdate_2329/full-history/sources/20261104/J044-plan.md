# J044：计时方法校准，无算子改动

JadeYara / PReluGradUpdate_2329。本轮唯一目标：定位J043顺序偏差的可观测来源，验证无profiler与必要profiler对照。允许个人J044测试入口、本题台账/报告/日志；不改J035/J038/J043源码/包，不碰提交页面、不提交、不push，不更改网络、安全、频率或系统配置。

原J021 runner逐次GetWorkspaceSize、launch、SynchronizeStream；J043每版独立进程，100次预热并未消除相同机器码控制的位置偏差。不能据Task减aiv_time命名根因。

本机CANN9 ACL1.17已声明EventElapsedTime、TIME_LINE和RICapture接口。官方9.0文档支持A3：
https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_0090.html
https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_0080.html
https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1782.html
事件计时输出毫秒，TIME_LINE自身有开销；捕获回放减少主机投递开销，但改变为连续执行条件，不能冒充官方单次kernel延迟。

阶段1：J038单输入FP32大目标pilot，比较sync、enqueue、capture-graph，32/128批长度，记录CPU wall和device事件区间；核对每阶段输出和捕获是否可用。只改新runner，按需编译，不安装。

阶段2：方法有效后先做J038同版本A/A，三dtype，小[2,3,1]、C32[127,32,1]、大[127,8192,1]、未改控制[600000,1,1]；变化顺序、预热、批长度。无profiler连续回放的事件span/次数作为热态吞吐口径，长度差分斜率用于检查固定边界开销。同一机器码A/A四轮大输入差应在0.10us或2%内才用于微小候选比较；失败则报告精度下限，不继续算法扫描。

阶段3：必要的少量匹配profiler/no-profiler对照，原样sync与graph，严格相同输入、预热和计时入口；分析Task/aiv_time/活动计数，保留全部原始数据。空闲频率只作空闲读数，不能推断单个kernel实际频率。若方法可辨识，再以J035/J038/J043未改包做有界对照；否则只给方法结论。

每次运行前检查旧作业；每阶段核对输出。优先原有输入/hash，不生成新算子代码。原始日志/CSV/JSON与来源指纹归档。
