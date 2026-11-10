# EXP_C005 — 本地六项Host拒绝原因已定位；平台Runtime Error仍未知

2026-11-10；hongwei-2026；唯一题目ConfusionTransposeD_2323。

## 平台事实

截图归档 `logs/EXP_C005/platform/platform-20261110-145611.png`；左侧日期被裁剪，可见时间14:56:11，按本会话日期归档。Owner确认平台不提供详细错误文本或失败shape。

- 总提交状态：**Runtime Error**。
- 可见点1～5：均Pass、输出错误0.00%。
- 点1：716.62us，最优3.66us；原1760us，约2.46×改善，仍慢约195.8×。
- 点2：728.01us，最优3.45us；原1750us，约2.40×改善，仍慢约211.0×。
- 点3：1.76ms，最优3.45us；原1.73ms，没有改善。
- 点4：1.84ms，最优3.50us；原1.70ms，没有改善。
- 点5：1.83ms，最优3.43us；原1.82ms，没有改善。
- 点6：隐藏；截图不足以确认Runtime Error发生在哪个点，不能认定一定是点6。
- 按会话上下文记为P004平台重提反馈；没有平台下载工程/源代码哈希，不能独立验证提交内容。

**结论：P004仅有部分平台收益，整次验收未过。** 保留EXP_C004同机22/22通过的事实，不将其扩大为平台功能全绿。

## 静态线索（不是根因结论）

1. Host沿用输入整tensor驻UB检查，超过UB返回GRAPH_FAILED；P004没有解决此前已登记的超UB债。
2. Host/InferShape沿用perm/attr shape同rank约束，合法的跨rank TF reshape会被拒绝；样例程序也曾先拒绝。TF=false的原输入rank可以不同，因而需分开探测。
3. attr reshape -1未做维度推导；Host的tiling函数存在空tensor拒绝分支。后续C005的空tensor探测两版均PASS_BITWISE，故不能将该静态分支等同于实际端到端拒绝。这些是否属于本题平台case未知。
4. 连续后缀路径仅对span>1生效；点3～5无收益支持覆盖不足的可能性，但没有tiling数据，不能确认其span。
5. 平台耗时与已知同机kernel数值差距大，需记录GetWorkspaceSize、launch和同步等主机阶段，不能直接把平台数值当已测kernel时间。

## P005 诊断计划

改Kernel前已登记P005（本轮不改Host/Kernel）。准备≤30分钟、首轮回传诊断≤30分钟；证据不足写DEC结案。

- 复用上一轮隔离目录内545f8d2与4761239精确Host/Kernel和已构建vendor包；逐源码核对，避免又编两套Kernel。
- 共52用例：原22项控制、15项极短tensor、5项此前未覆盖的整数存储类型、5档UB边界、跨rank TF/RF各1、-1属性TF/RF各1、空tensor1项。
- 诊断样例驱动仅放宽自己的crossrank/-1前置检查，让真实Host接到属性；每case独立进程，严格逐字节对拍，完整错误日志。
- 采集GetWorkspaceSize、launch、sync主机时间；**主机墙钟不是设备任务耗时**。不改变EXP_C004已归档profile结果。
- 每case超时30秒；进程信号/超时后该版本停止后续设备执行，余项SKIP，避免重复执行故障case。
- 拷贝安装CANN内Gather/Transpose相关头文件，只读供下一条路线核对910B可用API；本轮不实现Gather，也不接入950 SIMT。
- 诊断exit=0仅表示流程完成；每项FAIL_OR_REJECTED/TIMEOUT仍为失败，绝不把“预期拒绝”写成kernel PASS。

## 本地验证

- 52项fixture生成成功（**仅CPU生成，不是NPU PASS**）。
- 诊断C++源码锚点匹配通过；Python/Bash语法检查通过。
- 原 `local_semantic_check` 仍通过；本轮Host/Kernel未改。
- 下述云端执行已由Owner控制台粘贴回传；完整tar及逐case日志尚未取得。

决策见[DEC002](../decisions/DEC002-platform-runtime-error-diagnostic-first.md)。

## Q9AC 控制台回收（2026-11-10）

Owner无法下载tar，回传终端文本；原文逐字节保存在 `logs/EXP_C005/Q9AC/console-paste.log`，7093字节，SHA256 `bd877ba7ff55c6d52af50e2a4b621542c06c2c29d2ea3bb31245c3d9b126835a`。解析结果见同目录 `console-summary.json`。本轮**只有控制台摘要，不能称完整云日志已归档**。

控制台显示run为 `EXP_C005_20261110_153111_Q9AC`，源码核验完成、64份CANN头文件快照、local_semantic_check通过；baseline/candidate各52项，均46项PASS_BITWISE、6项FAIL_OR_REJECTED。两版逐项结果完全一致，未出现TIMEOUT或SKIP。此处源码核验仅有脚本成功输出，未取得其manifest用于本机独立复核。

- 两版均通过：原22项控制、15项极短tensor、5项整数存储类型、131072/163840字节UB探测、crossrank_rf、empty_tensor。
- 两版均未通过：`ub_188416_bytes`、`ub_196608_bytes`、`ub_262144_bytes`、`crossrank_tf`、`infer_tf`、`infer_rf`。
- `FAIL_OR_REJECTED`是脚本合并分类，既可能API拒绝，也可能运行/字节比较失败。没有逐case错误、退出码、DIAG阶段，不能断定具体分支、失败阈值或崩溃类型。
- 空tensor实测PASS_BITWISE应覆盖此前“端到端一定拒绝”的推测；不能由此证明Host tiling/Kernel实际被调用。
- 当前52项未观察到P004新增失败，**不等于证明P004对所有输入无回归**；两版同失败也不等于平台隐藏case根因已确定。
- 此粘贴没有host计时或新msprof数据；本轮不新增性能收益结论，diagnostic exit=0不代表52/52或平台验收通过。

按[DEC003](../decisions/DEC003-c005-console-evidence-insufficient.md)有限证据结案，继续冻结Host/Kernel。提供只读 `tests/print_c005_evidence.py` 导出已有逐case错误与代表项host阶段供复制，不重跑NPU；优先candidate，必要时补baseline全文。

本地补录：原语义检查通过；文本导出工具的Python语法、默认摘录标识、两版指定case全文、缺失case错误检查通过，记录在 `logs/EXP_C005/Q9AC/text-export-local-check.json`。这些是CPU工具验证，不是新的云端build/NPU回归或性能验收。

## 第二次文本回传：失败阶段已知，正文仍缺

原文 `logs/EXP_C005/Q9AC/detail-paste.log`（52713字节，SHA256 `42d2ffbbd42e16cc2b7ef54d3bd016f99f4c56e6af371add1b4a3343887900d0`）；结构化提取见 `detail-summary.json`。原诊断commit为9b6de1cb2fe06456972377b6b63b36bd21453844。

- 两版共12条失败记录均exit=2，阶段最后为`DIAG begin=GetWorkspaceSize`，没有成功完成该调用的标记，也没有begin=launch。结合诊断驱动的check/异常处理，问题发生在该调用范围内、尚未调用本算子的launch API；不属于已复现的Kernel执行阶段失败。具体返回码/触发原因尚未取得。
- candidate六份失败日志各约43.11MB。原导出工具只取首尾6000字符，均为初始化/清理消息，**中间错误被漏掉**；不能据摘录未见ERROR宣称没有错误。
- candidate代表case冷启动主机GetWorkspaceSize约1.49～1.54秒、launch API约0.88～1.04毫秒（empty约71微秒）。该诊断设置`ASCEND_GLOBAL_LOG_LEVEL=0`并输出海量调试日志，且每项独立新进程；这些数值不构成性能基准，不能与C004设备微秒profile或平台毫秒结果等同。
- empty_tensor的主机launch/sync API成功仍不证明实际Kernel发射；缺设备任务证据。

本次仅维护只读导出工具（≤15分钟），新增`--errors`扫描完整已有日志，保留错误/本算子原因/DIAG标记及前后文、原始行号、文件SHA256；超量匹配明确标记省略。原始完整日志和平台case仍缺失，不改Host/Kernel，不新增NPU试验或性能路线。

本地仅验证导出工具：超过43MB的模拟日志中间错误/ACL返回码能被提取，输出有界；大量匹配显式标省略，两版CLI运行通过。详见`Q9AC/error-filter-local-check.json`，不构成云端功能或性能验收。

## 第三次文本回传：本地原因结案

Owner回传全文错误筛选输出；原文`logs/EXP_C005/Q9AC/errors-paste.log`，61016字节，SHA256 `541617ff152733aff1336e86afe304ddb10d69877da98ac7b0184939dc310646`；结构化提取见`errors-summary.json`。六份candidate日志的报告字节数/哈希与第二次回传完全一致，ERROR匹配均total=6/retained=6/omitted=0。

1. `ub_188416_bytes`、`ub_196608_bytes`、`ub_262144_bytes`：tiling.cpp:159明确报`tensor too large for UB-resident scheme`。触发整输入字节数加8KiB预留达到/超过UB大小的检查；这是实现的主动限制，未证明发生设备内存越界或物理OOM。不能靠删除检查扩大标量回退的UB分配。
2. `crossrank_tf`：tiling.cpp:135明确报`perm/shape rank mismatch`。该case输入rank4、perm长4、最终reshape rank2；当前实现用attr shape rank约束perm，导致拒绝。
3. `infer_tf`、`infer_rf`：tiling.cpp:152明确报`shape numel mismatch`。当前直接乘attr维度，未解析唯一`-1`为5，因此未能与输入120元素匹配。

六项调用链均从tiling失败向上返回`ACL error 561002`，与先前exit=2、GetWorkspaceSize范围内且无launch的证据一致。561002在这里是共同上报码，具体原因以三条明确Host错误为准，不能把该码独自解读为某种设备故障。debug配置字段的JSON WARNING不作为根因。

baseline错误正文未回传；只能引用两版逐项分类/阶段一致、exit=2的证据，不能声称逐条baseline原错误已取得。通过Git静态核对，545f8d2与4761239的Host中均有这三类拒绝检查，故本轮未观察到P004新增失败，不能推广为所有输入无回归。

按[DEC004](../decisions/DEC004-close-c005-host-rejections.md)停止追加C005诊断。本次只归档/核对，不改测试工具或Host/Kernel；六项本地原因已解释。仍无平台失败输入、错误详情或提交源码哈希，因此**无法认定这三类边界就是平台RE原因**。未产生新性能数据，完整云端tar与原始全量log仍未回收，归档的是用户提供的筛选输出原文。
