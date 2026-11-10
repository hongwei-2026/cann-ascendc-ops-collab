# EXP_C005 — 平台重提 Runtime Error；P005 边界诊断待云测

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
3. attr reshape -1未做维度推导；空tensor被拒绝。这些是否属于本题平台case未知。
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
- 待Owner执行 `tests/run_c005_diagnostic.sh` 并回传 `EXP_C005_*.tar.gz` 后才能定位已复现边界；若没有复现，只记未知，不猜平台根因。

决策见[DEC002](../decisions/DEC002-platform-runtime-error-diagnostic-first.md)。
