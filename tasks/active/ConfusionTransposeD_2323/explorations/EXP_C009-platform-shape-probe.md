# EXP_C009 — Host诊断探针

2026-11-12；ConfusionTransposeD_2323；hongwei-2026；P009。

## 范围与假设

在P004的infer/tiling Host回调增加诊断输出，不改变判断、返回码、tiling、Kernel或输入边界。假设平台可回收包含`CTD_PROBE`的原始stderr；该假设尚未验证。按P009限额，本地实现及验证≤2小时，云端/平台回传分析≤1小时；超时写DEC。平台点6隐藏、既往平台无详细错误，不能预先假定日志可见、全部调用都会进入Host回调，或case与回调一一对应。

## 记录

`CTD_PROBE`后为一行JSON，包含进程号、阶段内序号、时间、infer/tiling阶段、输入维度、numel、perm、shape属性、transpose_first；tiling还记录`dtype_code`（GE数值枚举）。infer成功记录输出shape；tiling通过UB检查后记录实际copy_span、连续行/标量路径、block_dim和UB tile。UB检查失败记录UB与输入字节数。每个回调入口和退出各一行，退出有`host_ok`和最后检查`stage`；该值只说明对应回调成功或失败，不等于外部ACL错误码或设备launch成功。实际错误码及全文取调用方/框架原始日志。

入口与退出通过`pid`、`phase`、`seq`配对；不同阶段的序号各自独立，框架缓存、并行执行或进入回调前失败时不能推断平台case编号。实际输出、平台能见度、launch和平台点对应均需证据。探针向stderr写入，改变Host开销，探针版本性能不能当P004或TBE性能基准。

## 验证与下一步

本地`local_semantic_check.py`和P004搬运计划1658项检查通过；Git Bash检查云测脚本语法通过；用最小Host数据结构编译`c009_probe_host_smoke.cpp`并解析4条JSON，确认成功/失败事件字段。该模拟不验证CANN ABI、平台日志或NPU结果，详见[local-verification.log](logs/EXP_C009/local-verification.log)。

CodeArts按[C009-CLOUD-RUN.md](../C009-CLOUD-RUN.md)隔离构建P004和探针候选，各运行`build.sh -e`和39项bitwise回归。候选另运行C005边界探测，并验证正常fixture的infer/tiling记录和UB拒绝记录。云端未执行前不称构建或精度通过；平台未提供日志前不称拿到任何平台shape。云测完成后才能评估探针包是否用于平台；每个已知shape的TBE profile待平台真实输入可取得后开展。
