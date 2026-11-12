# EXP_C009 — Host诊断探针

2026-11-12；ConfusionTransposeD_2323；hongwei-2026；P009。

## 范围与假设

在P004的infer/tiling Host回调增加诊断输出，不改变判断、返回码、tiling、Kernel或输入边界。假设平台可回收包含`CTD_PROBE`的原始stderr；该假设尚未验证。按P009限额，本地实现及验证≤2小时，云端/平台回传分析≤1小时；超时写DEC。平台点6隐藏、既往平台无详细错误，不能预先假定日志可见、全部调用都会进入Host回调，或case与回调一一对应。

## 记录

`CTD_PROBE`后为一行JSON，包含进程号、阶段内序号、时间、infer/tiling阶段、输入维度、numel、perm、shape属性、transpose_first；tiling还记录`dtype_code`（GE数值枚举）。infer成功记录输出shape；tiling通过UB检查后记录实际copy_span、连续行/标量路径、block_dim和UB tile。UB检查失败记录UB与输入字节数。每个回调入口和退出各一行，退出有`host_ok`和最后检查`stage`；该值只说明对应回调成功或失败，不等于外部ACL错误码或设备launch成功。实际错误码及全文取调用方/框架原始日志。

入口与退出通过`pid`、`phase`、`seq`配对；不同阶段的序号各自独立，框架缓存、并行执行或进入回调前失败时不能推断平台case编号。实际输出、平台能见度、launch和平台点对应均需证据。探针向stderr写入，改变Host开销，探针版本性能不能当P004或TBE性能基准。

## 首次云测回传（2026-11-12，进行中）

- 代码 `ed583cb8610694ff87620865dbf8de93bfa0fb83`；P004基线及候选 `build.sh -e` 均样例PASS（0/120），双方各39/39 bitwise PASS。候选边界46 PASS、6 FAIL_OR_REJECTED；这是本地构造用例，不是平台测试点。
- 脚本整体 `exit=1`，停在 `probe evidence check`。39项均有匹配的成功tiling事件，但校验器要求的成功infer事件均为0；只证明日志中未观察到，不能推断框架是否执行infer，也不能称完整探针验证通过。
- 粘贴的检查器摘要中，`ub_196608_bytes` 在tiling报告 `tensor_too_large_for_ub`，`crossrank_tf` 报 `perm_shape_rank_mismatch`，`infer_tf/rf` 报 `shape_numel_mismatch`。这些是本机构造的拒绝，不能归因于隐藏点的Runtime Error。
- [控制台回传](logs/EXP_C009/C009-console-evidence.log)仅含打印的摘要和build日志末尾，SHA256 `aabcece950cc755c2f7e2612b1ea8915fdb6a65f05d04b0380de5caf4c4b0334`。完整tar尚未回收，云端路径为 `/mnt/workspace/problems/ConfusionTransposeD_2323/explorations/EXP_C009_20261112_181640_Og8y.tar.gz`。不得把摘要当作完整原始日志。
- 下一步复用这次云端已有逐case日志和CANN头文件，核实成功事件的实际字段、输出shape接口和infer日志缺失的可观测范围；如需改探针，重新编译和实机回归。平台stderr可见性、实际shape、隐藏点失败阶段仍未知；不把infer缺失改判为PASS。

## 先前准备与验证

本地`local_semantic_check.py`和P004搬运计划1658项检查通过；Git Bash检查云测脚本语法通过；用最小Host数据结构编译`c009_probe_host_smoke.cpp`并解析4条JSON，确认成功/失败事件字段。该模拟不验证CANN ABI、平台日志或NPU结果，详见[local-verification.log](logs/EXP_C009/local-verification.log)。

CodeArts按[C009-CLOUD-RUN.md](../C009-CLOUD-RUN.md)隔离构建P004和探针候选，各运行`build.sh -e`和39项bitwise回归。候选另运行C005边界探测，并验证正常fixture的infer/tiling记录和UB拒绝记录。云端未执行前不称构建或精度通过；平台未提供日志前不称拿到任何平台shape。云测完成后才能评估探针包是否用于平台；每个已知shape的TBE profile待平台真实输入可取得后开展。
