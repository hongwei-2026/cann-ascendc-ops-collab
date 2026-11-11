# EXP_C006 — P006末维不连续的步长DMA候选

2026-11-11；ConfusionTransposeD_2323；hongwei-2026。Owner明确授权继续优化；先登记P006/SESSION，再改Host/Kernel。基线为P004=4761239，沿用910B/CANN9.1.x。

## 可检验假设与预算

P004仅对连续后缀>1收益；同机rank2标量回退8.32us，未改善。P006改用已在P004验证过的DataCopyPad系列接口，增加GM规则步长，避免span=1布局的逐元素GetValue/SetValue。假设较大span=1布局对P004≥10×，原rank2降低≥2%；这些目标**尚未实测**，不能据CPU对照宣称收益，也不能将新probe等同平台未知case。

准备/实现/CPU验证≤90分钟，首轮云回传分析/修复≤30分钟，等待Owner不计入主动时间；小shape主动分析累计≤2小时，触止损写DEC。首轮准备约20分钟以内；不在等待期间再换路线。

## 本次变化

- 实际转置输入由TF决定：TF=true用原输入，TF=false用attr shape。连续后缀为1时，取最后非单元输出轴的长度和对应源stride，写入gatherSpan/gatherStep；两个字段均用于真实分派/搬运，dtype key保持原映射。
- MTE2每次按blockCount读取一个规则源序列，每块1元素；GM的srcStride按字节填写，UB每块自动补齐32B。跨输出轴进位或tile边界时重新MapSrcIndex，数据本身不经标量寄存器。
- MTE3从每个32B槽取有效1元素密集写回。每个tile至多4095槽；Host按槽数×32B计算真实临时UB，保留8KiB；核起点与非尾tile输出起点对齐32B，避免核间写同一GM块。
- P004连续后缀路径保持原算法；不能证明规则stride的情况保留原路径。没有接入Gather/Transpose新API或950 SIMT。
- 当前整输入大小上界检查仍保留，跨rank TF/-1边界也未改；因此**P005三类Host拒绝没有被修复，平台RE仍未知**。这是性能候选，不是功能债扩展。

## 已完成的本地验证

`logs/EXP_C006/local/verification.json`和`fixture-results.json`记录结果；大型fixture生成文件仅本地保留，不入Git。

- 原local_semantic_check通过（8布局及reference dtype检查）。
- P004 NumPy搬运计划1658项通过。
- P006独立NumPy字节oracle共1002项步长计划通过：不同perm/rank/TF/RF/同rank reshape、1/2/4/8字节、短尾、单元轴、UB槽容量、4095块上限及跨tile；检查源地址在界、UB在界、输出每元素恰写一次、核起点32B对齐。
- 云测39项fixture全部生成，含bf16，无dtype跳过；结果为GENERATED_ONLY，**未执行NPU**。
- Python/Bash语法通过；终端报告工具合成日志检查通过，保留真实错误、排除mismatched=0误报，失败profile不生成加速比。

CPU模型不能验证硬件的srcStride语义、队列同步、编译支持或性能，必须运行下述云测。此时没有CANN build/NPU PASS，也没有P006速度数字。

## 云端验收准备

`tests/run_c006_cloud.sh`从Git精确导出P004/候选到新的隔离目录，逐版本build.sh -e、同一驱动/输入的39项逐字节回归；每case独立进程、60秒超时，设备异常后余项SKIP且整体非零，不把SKIP算通过。

39项=原22控制+17步长布局。覆盖131072字节的f32/f16/i8/i64转置、奇数边长、短尾、尾单元轴、RF原输入跨rank、rank8及180096字节i8强制跨UB槽tile。源代码/完整构建和错误日志随tar归档，源码另存SHA256清单。DEBUG洪水关闭，错误保留。

同机msprof共7项：原rank2、4种较大span=1 dtype、样例连续行、128元素连续行；每项25次识别到的kernel，丢前5取余20次中位数，保留min/max、全部样本、Block Num和原CSV。只有执行成功、精确字节一致、kernel数正好25且时间有效时才标PROFILED；无msprof或未知格式不能用host墙钟代替。

结束或失败时直接打印`C006_EVIDENCE_BEGIN/END`，同时把原包复制到CodeArts项目文件树的个人本题`deliverables/cloud-logs/`，便于下载。仅粘贴摘要不等于完整云日志已归档，回传后如实记录证据范围。

下一闸：Owner执行[C006-CLOUD-RUN.md](../C006-CLOUD-RUN.md)，回传紧凑终端输出/原包；在30分钟预算内按结果采纳或DEC，不预先推荐平台提交候选。
