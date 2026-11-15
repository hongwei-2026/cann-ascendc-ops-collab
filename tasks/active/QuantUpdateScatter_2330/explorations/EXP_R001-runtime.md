# R001 — ACL 161002 本地修复（2026-11-15）

## 用户报告

第 1 点 Pass（5.89us），第 2 点 Runtime Error，后续 Skipped；returncode=2，ACL error 161002。另有 msprof --export 在指定 --application 时无效的 warning。没有完整 plog、实际失败 API 名、第 2 点 shape/dtype/attrs 或上传工程版本，不能断言平台失败根因已复现。

## 证据与修复

官方 ACLNN 公共返回码定义：161002 = ACLNN_ERR_PARAM_INVALID；可通过 aclGetRecentErrMsg 查看细节。官方 aclnnInplaceQuantScatter 约束列出 int8 var 下四种组合：fp16/fp32/int32 与 bf16/bf16/bf16 两组三元组，各自可用 int32/int64 indices。

本地 OpDef 原来只有两个签名列：fp16+int32 indices 和 bf16+int64 indices，缺少另两种；原 kernel 也固定绑定索引类型，虽然 tiling 已记录 indicesIsInt64，却没有用于 kernel 分发。该缺口确定存在，是本轮修复对象；是否对应平台第 2 点仍须重测确认。

- OpDef 补齐四个配对签名，format/unknown format 列数同步。
- 保留两种 updates dtype 的 mode/key；内核按已有 indicesIsInt64 选择 int32/int64 模板，修复“只改注册仍误读索引”的风险。
- Tiling 明确拒绝非 int32/int64 indices。
- 示例错误检查补 API 表达式和立即获取的 aclGetRecentErrMsg；保留 returncode=2 约定。
- CPU 模型对拍扩展到四种签名；CANN host/kernel UT 补上两个缺失组合及 inplace/no-zp 组合。
- 不改变 Q001 的量化舍入路径、SyncAll 或其数据调度。

## 官方来源

- https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnInplaceQuantScatter.md
- https://www.hiascend.com/doc_center/source/zh/canncommercial/81RC1/apiref/aolapi/CANN商用版 8.1.RC1 算子加速库接口参考 01.pdf （公共接口返回码，第 54 页）

## 验证状态

- 9 项 CPU 回归通过，现有调度对拍扩展为 1152 组，覆盖四种 dtype 组合。
- 用本地 C++17 编译器对实际 OpDef、kernel 入口和 sample_support.h 做源码契约检查：四个配对签名、四种索引分发、API 名/详细错误保留均通过。只替换 CANN API 及计算类为最小桩；这不是 CANN 编译，也没有执行真实设备内核。
- 原参考自带探针完成，全部 Python 源 AST 解析完成，确认 R001 未改 Q001 计算/同步头文件。
- 完整 stdout/stderr 日志：logs/EXP_R001/local-regression.log、source-contracts.log、reference-probes.log；摘要与源码 SHA256 同目录。
- 设备编译、精度和该平台第 2 点重测均 NOT_RUN。原始用户报错文本已归档，未冒充完整设备日志。
- 结论：已修复一个确定的官方合法输入覆盖缺口；平台第 2 点的具体根因仍未确认，需新工程重测或完整 ACL detail。
