# J067：原生FP16性能正证据；发现真实TBE契约差异，暂不宣称完全合规

## 结论
独立从J035起步，不含J065/ShortS。原生FP16 Mul/Muls、Compare、Select取代往返FP32 Cast，Host/keys/分核/tile保持，half比较及队列容量统一128元素粒度。FP32/BF16旧数学保留，30/30函数体逐字节相同；A/C冻结J03545/45函数相同；标准候选/B45/45函数相同。

- 实际同一可信910C/CANN9.3：ABC与标准all/binary/package通过。
- 原970组严格精度：ABC4026调用、标准1342调用全过。
- 12组新增边界：六路由全65536个FP16特征位模式，配最大有限及混合极值g/w，含subnormal/±0/NaN/Inf；ABC72、标准24调用全过。沿原严格bit逻辑检查，仅原有±0/NaN类别等价，不用1e-3掩盖错误。
- 性能94输入、2256窗口、789600完整Task；六种平衡顺序，每窗口350调用弃前200保留150，固定五tensor和流。
- 25个FP16输入汇总25/25提升，150/150轮快于A/C；范围1.24–17.59%。FP32/BF16无启用新数学路线。
- [127,8192,1] 11.940→9.84025us（-17.59%，少2.09975us）；[4097,128,1] 10.29525→8.780（-14.72%）；[127,64,3]89.922→81.382（-9.50%）；[127,64,1]6.460→6.380（-1.24%）。不推断官方shape，不保证官方增益。

## 控制未全过，未重跑刷门槛
559/564严格控制通过，5次失败原样保留：FP16[257,257,3]两次0.310/0.319us控制幅度、FP16[127,1024,3]一次0.139us量级；FP32[4097,1,1]2.49%左右相对控制差；FP32[257,8192,1]一次0.10000000000000142us浮点边界。控制幅度取原A1/A2/B/C的max-min，原值保存在measurement-summary.json和CSV，没有舍入门槛、放宽或复跑。

## 性能机制与验收边界
同轮[127,8192,1]的A1/A2 Vector活跃中位数3.850us，B1.999us；MTE2约3.69→3.701us，支持本次数值通路减少Vector工作，而不是改变GM数据量。引擎时段交叠，不相加解释Task收益。两次内置TBE探针仅做语义，不是性能对标，尚不能据此宣称本候选达到所有合法输入的TBE95%性能目标。

## 首轮失败和修正
Attempt1原生half直接沿用FP32的64元素对齐；channel-edge400元素被补到448half，896B不满足Compare接口256B约束，正features在local384误选负支。首轮停止精度后性能未启动，完整源码/日志/机器码已保存。R2统一half比较与Init缓冲128粒度，原失败及全位模式通过。不能只改cmpCount而不扩缓冲，否则tileLen64会越界。UB各路径上界小于188416B。官方约束见reassessment里的Compare链接。

## 更重要的TBE泛化/精度事实
取回真实安装TBE发现ND/NCHW/NC1HWC0、grads/features广播、高rank空间weights，当前工程仅ND、同形输入、rank1通道weights。这些合同尚未补齐。
已用公开aclopCompileAndExecute且unset自定义OPP，成功实测内置TBE两种dtype各64元素。FP16有限x=-1/g65504/w2输出dxInf/updateNaN；FP32最大有限值同理。实际dx/update各64/64匹配TBE源码DAG，update各32/64不同于模板where。TBE实际还有量化dx后乘0传播操作，而当前J035/J067 where-golden没有。因此本轮精度通过只证明reference/J035兼容，不能宣称全部TBE边界兼容，不能因平台五点Pass忽略合法溢出场景。
TBE合同探针最初编译头路径错误已修；两次NPU调用成功后脚本末尾CRLF导致后处理NameError，输出已独立校验，没有重跑NPU或隐藏错误。全日志在单独TBE证据包。

## 文件及下一步
- attempt2/candidate-standard-source.zip：65209B，46文件、标准code/根，SHA256 8876c423f900e58f909df00cb97b3a4239c25f7bf54e85662bab621b37ec23cf。是实验源码，不是已全TBE合规的提交建议。入口与五头必须配套，入口恢复J035六参Init，不可混J065七参入口。
- attempt2/device-evidence.zip：32608354B、138文件，SHA256 dd3d0d2af57fcb633ce0281fa8bc8e7c82d679f626e08c092cf01e852e8222a5，16小块及全包SHA/全部CRC通过后原子发布，可完整读取，未再写入。
- tbe-contract-evidence.zip：14291B/36文件，SHA2567326e53da2cb780e1c610df2691eea7f90a5a6d3a0c1586cf14e77b8872bded0。
- reassessment-20261107.md：官方成绩比较、真实契约、未探索结构排序。
优先把TBE真行为建立为独立oracle并明确与模板差异，再做任意C/中等S连续块+批量向量权重广播；保留原生half性能结论，在完整TBE系数链下重新验证。J066继续暂停，生产J03546文件未动，无官方操作、commit/push。
