# EXP Q005 — Q002 瓶颈采集准备（本地）

日期：2026-11-15；起点：2d3c61c。承接 DEC_Q003 回退结论，先登记 DEC_Q005-profile-first 和 path-ledger 新行，再实施本轮工具。

## 结果

新增独立诊断 runner、26 组确定性输入生成/逐字节校验、msprof 两种入口的能力检查、原始数据归档和保守汇总。默认 10 次预热 + 50 次采样、3 轮交替 case 顺序。同址/异址诊断配对共用同一输入和 golden；场景含大拷贝、窄多行、宽行、多 batch 和四 dtype 签名。它们是人工诊断输入，不是官方 50 点。

生产 op_host / op_kernel、默认 sample 源码均未改，TQue 深度仍为 2，InitBuffer 两槽、SyncAll、fp32→int16(RINT)→half→int8 保持 Q002。

## 文件

- workdir/code/examples/profile_quant_update_scatter.cpp：独立参数化 ACLNN runner，每次执行重新获取 executor；可选 zp、reciprocal、同址输出和重复运行。
- workdir/code/examples/CMakeLists.txt：新增 EXCLUDE_FROM_ALL 的 profile_case 目标；普通样例构建不会编译该目标。
- workdir/code/examples/profile_q005.sh：独立构建 runner 后直接采集，构建在采集范围外；新目录防覆盖。
- workdir/code/tools/profile_q005.py：输入/golden 哈希、case 参数、源/包/程序标识、完整 collector 日志、逐轮输出及结果归档。
- workdir/code/tools/README_Q005.md：完整运行方法、结果解释及限制。
- workdir/code/tests/local/run_q005_support.py：用同步 ACL stub 编译并执行真实 runner 的本地检查。

## 测量口径

优先使用镜像支持的 application 模式；不支持时检查 msprof op。两种 CLI 的帮助信息和每次完整命令均归档，不同时传 --export 和 --application，不把编译过程当作应用性能。

op_summary 仅在目标任务数与预期调用数完全相等、时间戳有效且唯一、时长为正时，按时序剔除预热后汇总任务时长。PipeUtilization/OpBasicInfo 每核行不当作 launch 样本。只存在原始表时标 RAW_ONLY，不生成时长或加速结论；空表或缺表标 INCOMPLETE。校验结果和测量完整度分开记录。

## 本地验证

- 26 组诊断数据与辅助检查通过：输入/黄金哈希、错误输出、时间排序预热剔除、缺失/不完整/空导出、带空格路径与采集参数。
- 实际 runner 编译及 CPU ACL 接口模型：26 组 × 3 次调用精确对齐 reference；覆盖四签名、可选 zp、reciprocal、同址输出、workspace 扩容和错误 index_bits 拒绝。
- R001 实际源码契约检查：四签名、四 dispatch、错误调用名和 ACL detail 通过。
- shell 语法检查及 git diff --check 通过（Git 提示未来行尾转换，不是检查错误）。
- 旧 run_q002_kernel_model.py 报 Saved Q002 baseline required，因主仓缺少其历史备份；本轮未修补或伪称通过，失败全文保留。

日志：explorations/logs/EXP_Q005/actual-runner-cpu.log、r001-contracts.log、local-support-checks.log、q002-kernel-model.log、local-doc-reference.log。

本地官方 CANN 8.5.0 开发指南 PDF 第 456 页（正文 422）提供 msprof op 及 PipeUtilization 采集方式，第 479 页（正文 445）强调 MTE2/MTE3 时间；节选及文档位置已留存。实际镜像 CLI 仍以现场帮助信息为准。

## 决策与下一闸

本地工具准备完成；设备证据待采集，生产继续 Q002。CANN 编译、NPU 精度、profile 实采、同机标杆与新增性能收益全部 NOT_RUN。

**Cursor review（同日）**：采纳为诊断工具；生产代码未改；已 push main。

**云采（同日 `@d8ccf35`）**：`copy_heavy` COMPLETE。

| case | overall median (us) |
|------|---------------------|
| out（异址） | **29.62** |
| in（同址） | **9.10** |

Δ≈20.5 µs → 该诊断上整表拷贝主导异址耗时。详见 `logs/EXP_Q005/copy01-medians.json`。平台 50 点 inplace/shape 未知，**不**据此立刻改拷贝或流水。下一步：导出 aiv_mte2/mte3/vec 中位；采 narrow_rows / wide_rows。
