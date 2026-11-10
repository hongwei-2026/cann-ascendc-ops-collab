# J068：真实TBE数值修复已验证，性能回退，不晋升

## 已完成的实际结果
同一已核验910C，CANN9.3，目标ascend910_93。独立J035起点，五Compute改为TBE实际系数/负mask乘法链；half采用原生half且Compare/队列128粒度，FP32/BF16保留FP32计算；dx先量化后乘0加到update。Host、key、分核、tile和现有ND格式注册保持，生产46文件哈希不变。

- ABC和标准all/binary/package通过，标准与B45/45函数体一致。
- 27组直接内置TBE二进制oracle，含三dtype、六路由、全FP16/BF16特征位模式与非有限/溢出，别名候选54次、标准54次严格通过。
- 旧970组中9组where-golden与TBE不同，另961组完整DAG与旧golden一致。只为9组另取真实内置TBE输出并保存到新目录；旧golden从未覆盖。修复B和标准各970组/1342调用全部通过。
- 原旧回归在[8,64,3]元素18停止，g0/x2/wInf，旧dx=0；同输入直接内置TBE的dx=0x7fffffff NaN，与新核一致。原失败日志保留，不冒称旧套件原样全过，也未用误差阈值掩盖差异。
- 94个性能输入经完整DAG扫描无新旧语义差异，可与冻结A/C公平比较；2256窗口、789600调用，六平衡顺序/固定五buffer和stream/350弃200，控制562/564。

## 净性能结论
J068是正确性基线，不是提速提交候选。94输入中：
- FP32：44个，0个输入/0轮获益，逐输入收益中位数-12.88%。
- FP16：25个，9个输入/54轮获益，逐输入收益中位数-0.36%。
- BF16：25个，0个输入/0轮获益，逐输入收益中位数-10.69%。
[127,8192,1]：FP32 10.92275→13.601us（慢24.52%），FP16 11.7605→11.46（快2.56%），BF16 12.02→16.161（慢34.45%）。不推断官方shape，不把修复包装成优化。不能靠恢复错误where语义换性能。

## 实际合同探针
公开ACL内置PReluGradUpdate，unset自定义OPP：ND通道权重、两方向grads/features广播、空间weights、NCHW通道/空间weights、BF16、rank8、N/C/S三个空维、三dtype特殊值均执行成功且匹配DAG。NC1HWC0原始NCHW[2,19,3,5]/物理[2,2,3,5,16]，修正origin/storage描述后，三dtype各570逻辑元素的dx/update全部零差异。
首轮NC1仅设置origin，遗漏独立storage字段，输出不能作为NC1语义结论；其全证据保留在contract-attempt1-evidence.zip（43702B/162文件，SHA1c19d52ee51a15f8bd585d26eaaa0357c0175a4ad20fc45c9d10f97f8d3df849）。正确顺序见design/REPAIR_AND_GATHER.md及contract/probe-fixed.cpp。
这些是实测具体合同点，不是全笛卡尔积证明。未知rank图编译、更多混合格式、特殊非标准weights形式和全部合法输入的TBE95%性能仍未确认。

## 泛化修复设计与新结构
现有InferShape要求同形并输出features；tiling拒C0/S0；OpDef只发布ND，权重只rank1。真实合同需要广播输出形状、normalize三源stride、空间weights变形、分离logical/storage及空输出no-op。
design/broadcast_layout_model.py已实现可执行的Host/内核描述模型：广播、权重变形、连续轴合并、source offsets和精确整数division。对17探针的全部输出位置验证原始位索引；全magic50,331,648局部坐标验证无错。这是CPU模型，不冒称NPU泛化执行器已实现。
连续块Gather方案保留全局32B分核，一次处理任意C/中S连续tile，批量权重窗口/整周期缓存，纯整数Muls+ShiftRight生成索引，跨尾第二DMA用32B对齐洞和索引修正。源码/API可用性及UB/同步/路由门槛/测试矩阵在design/REPAIR_AND_GATHER.md。
目标SDK最小Gather编译仍未执行；用户已明确确认本项目传输许可；操作获准后遇到官方转发kex关闭，正在恢复，仍不得称目标编译失败或已通过。当前云端大证据包尚未生成/下载。原始日志、二进制oracle和CSV留在同一云实验J068目录。本地已有源码、首轮合同证据和设计；未官方提交、合并、commit或push。
