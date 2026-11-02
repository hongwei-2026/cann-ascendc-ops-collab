# SpMV算子开发任务书

## 基础信息

- **技术标签**：算子开发
- **适配硬件**：Ascend 950PR
- **开源仓地址**：https://gitcode.com/cann/ops-sparse
- **CANN 版本**：算子开源仓指定版本
- **开发语言**：Ascend C

## 1. 任务概述

参考[ cuSPARSE SpMV 实现](https://docs.nvidia.com/cuda/cusparse/#cusparsespmv)，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的稀疏矩阵向量乘算子，完成算子设计、开发、测试全流程工作，验收通过后将算子提交至昇腾算子开源仓。

SpMV 计算公式：
$$Y = \alpha \cdot op(A) \cdot X + \beta \cdot Y$$
其中：
- $op(A)$ 稀疏矩阵（CSR 格式）
  - 当开启 transpose 时：$op(A) = A^T$（矩阵转置）  
  - 当不开启 transpose 时：$op(A) = A$（原矩阵）
- $X$ 为稠密输入向量
- $Y$ 为输入输出稠密向量
- $\alpha$、$\beta$ 为标量系数

## 2. 核心开发要求

### 2.1 功能实现要求

与 cuSPARSE SpMV 核心功能完全对齐，支持 CSR 稀疏数据格式，支持的数据类型组合如下：

| 输入 A、X 类型 | 计算类型 (computeType) | 输出 Y 类型 |
|---|---|---|
| float32 | float32 | float32 |
| int8 | int32 | int32 |
| int8 / float16 / bfloat16 | float32 | float32 |
| float16 | float32 | float16 |
| bfloat16 | float32 | bfloat16 |

**数学公式**：

$$Y = \alpha \cdot op(A) \cdot X + \beta \cdot Y$$

其中：
- $op(A)$ 为稀疏矩阵（CSR 格式），当开启 transpose 时 $op(A) = A^T$，否则 $op(A) = A$
- $X$ 为稠密输入向量
- $Y$ 为输入输出稠密向量，支持原位累加
- $\alpha$、$\beta$ 为标量系数

**算法说明**：

1. **CSR 格式稀疏矩阵**：稀疏矩阵 A 通过三个数组表示——`csrRowPtr`（行偏移，长度 M+1）、`csrColInd`（列索引，长度 NNZ）、`csrVal`（非零元素值，长度 NNZ）。NNZ 为矩阵非零元素总数。
2. **转置处理**：当 trans=true 时，对稀疏矩阵 A 执行转置操作后再参与计算，即 $op(A) = A^T$，矩阵维度从 M×K 变为 K×M。
3. **三阶段调用流程**：与 cuSPARSE 接口对齐，分三阶段完成调用：
   - 阶段 1：获取 workspace 大小（GetBufferSize）
   - 阶段 2：预处理（Preprocess，实际调用时可选，用于优化后续计算性能）
   - 阶段 3：算子执行（SpMV 计算入口）
4. **原位累加**：Y 向量既是输入也是输出，通过 $\beta$ 系数控制是否累加原有 Y 值。当 $\beta = 0$ 时相当于纯矩阵向量乘。
5. **稀疏度泛化**：需支持不同稀疏度（50%~99.9%）的输入矩阵，稀疏度越高 NNZ 越小。

**确定性计算要求**：算子需支持确定性计算，相同输入多次执行结果一致。

### 2.2 算子工程模式

使用 Ascend C 的算子工程化开发方式进行算子开发，接口设计参考 cuSPARSE 原生头文件 `cusparse.h`，采用 aclsparse 接口风格封装。

### 2.3 接口定义

```C
/**
 * @brief 获取稀疏矩阵向量乘法(SpMV)所需的缓冲区大小
 */
aclsparseStatus_t aclsparseSpMVGetBufferSize(aclsparseHandle_t           handle,
                                             aclsparseOperation_t       opA,
                                             const void                *alpha,
                                             aclsparseConstSpMatDescr_t matA,
                                             aclsparseConstDnVecDescr_t vecX,
                                             const void                *beta,
                                             aclsparseDnVecDescr_t      vecY,
                                             aclDataType                computeType,
                                             aclsparseSpMVAlg_t         alg,
                                             size_t                    *bufferSize);

/**
 * @brief 对稀疏矩阵进行预处理,加速后续 SpMV 计算
 */
aclsparseStatus_t aclsparseSpMVPreprocess(aclsparseHandle_t           handle,
                                          aclsparseOperation_t       opA,
                                          const void                *alpha,
                                          aclsparseConstSpMatDescr_t matA,
                                          aclsparseConstDnVecDescr_t vecX,
                                          const void                *beta,
                                          aclsparseDnVecDescr_t      vecY,
                                          aclDataType                computeType,
                                          aclsparseSpMVAlg_t         alg,
                                          void                      *externalBuffer);

/**
 * @brief 稀疏矩阵向量乘法(SpMV)计算入口
 */
aclsparseStatus_t aclsparseSpMV(aclsparseHandle_t           handle,
                                aclsparseOperation_t       opA,
                                const void                *alpha,
                                aclsparseConstSpMatDescr_t matA,
                                aclsparseConstDnVecDescr_t vecX,
                                const void                *beta,
                                aclsparseDnVecDescr_t      vecY,
                                aclDataType                computeType,
                                aclsparseSpMVAlg_t         alg,
                                void                      *externalBuffer);
```

### 2.4 参数说明

以下仅罗列算子底层原始输入参数，实际开发实现过程中，参考 cuSPARSE 接口设计，将稀疏矩阵相关入参封装为统一描述符结构（`aclsparseConstSpMatDescr_t`）进行传参使用。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | dtype类型 | 数据排布格式 | 维度(shape) | 值域范围 | 异常行为 |
|---|---|---|---|---|---|---|---|---|
| csrRowPtr | 输入 | 稀疏矩阵 A 行偏移数组，存储每行非零元素起始位置 | tensor | INT32 | 一维数组 | [M+1] | [0, NNZ] | 触发参数校验报错 |
| csrColInd | 输入 | 稀疏矩阵 A 列索引数组，存储非零元素所在列号 | tensor | INT32 | 一维数组 | [NNZ] | [0, K-1] | 触发参数校验报错 |
| csrVal | 输入 | 稀疏矩阵 A 非零元素值，存储稀疏矩阵有效数值 | tensor | FLOAT16、BFLOAT16、FLOAT32、INT8 | 一维数组 | [NNZ] | (-∞,∞) | 触发参数校验报错 |
| x_vec | 输入 | 稠密输入向量 X，参与矩阵向量相乘计算 | tensor | FLOAT16、BFLOAT16、FLOAT32、INT8 | 一维 | [K] | (-∞,∞) | 触发参数校验报错 |
| y_vec | 输入输出 | 稠密输出向量 Y，承载最终计算结果，支持原位累加 | tensor | FLOAT16、BFLOAT16、FLOAT32、INT32 | 一维 | [M] | (-∞,∞) | - |
| trans | 属性 | 矩阵转置标识，控制是否对 A 进行转置计算 | scalar | bool | - | - | True/False | 触发参数校验报错 |
| alpha | 属性 | 前置缩放系数，缩放矩阵向量相乘结果 | scalar | FLOAT32 | - | - | (-∞,∞) | 触发参数校验报错 |
| beta | 属性 | 后置缩放系数，缩放原有 Y 向量数值 | scalar | FLOAT32 | - | - | (-∞,∞) | 触发参数校验报错 |
| compute_type | 属性 | 中间计算精度，指定计算过程数据精度 | scalar | INT32、FLOAT32 | - | - | - | 触发参数校验报错 |

**参数约束**：

1. csrVal、x_vec 的数据类型需与数据类型组合表匹配，compute_type 需对应指定；
2. 当 trans=false 时，矩阵 A 维度为 M×K，x_vec 长度为 K，y_vec 长度为 M；当 trans=true 时，矩阵 A 维度为 K×M，x_vec 长度为 M，y_vec 长度为 K；
3. csrRowPtr 须为单调递增数组，csrColInd 须在有效列范围内；
4. 所有输入 tensor 支持非连续 Tensor。

## 3. 验收标准

### 3.1 软硬件环境要求

- **适配硬件**：Ascend 950PR
- **CANN 版本**：算子开源仓指定版本

### 3.2 精度要求

1. 算子计算精度需满足[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)；

### 3.3 性能要求

1. 算子整体性能需达到 0.5 倍 标杆水平；
2. 性能自测 case 如下：

| 编号 | 矩阵规模 | 稀疏度 | 数据类型 | 标杆性能 |
|---|---|---|---|---|
| 1 | 128×128 | 95% | float32 | 43.9us |
| 2 | 1024×1024 | 99% | float32 | 46.3us |
| 3 | 2048×4096 | 97.5% | float32 | 45.4us |
| 4 | 160220×68750 | 99.9% | float32 | 193.2us |

### 3.4 内存要求

不涉及

### 3.5 自验要求

本任务已提供[测试用例](./spmv_testCase/)，验收阶段将采用泛化数据进行功能、精度、性能全维度验证，可自行设计脚本测试该用例。自验证报告完整、可复现，所有测试用例执行通过。

测试用例覆盖**常规场景、边界场景、转置/非转置、layout 行/列优先、全数据类型、alpha/beta 组合、稀疏度覆盖 50%~99.9%** 等所有功能场景。

本任务的自测用例的输入数据生成遵循如下分布规则，如果需要进行更完善的自测，请参考如下规则生成自测用例：

| 参数名 | Tensor值域分布 | Attr 覆盖规则 |
|---|---|---|
| csrVal | (-10, 10)均匀分布，100% | FLOAT16、BFLOAT16、FLOAT32、INT8 全覆盖 |
| x_vec | (-10, 10)均匀分布，100% | 同 csrVal |
| y_vec | (-10, 10)均匀分布，100% | 同 csrVal |
| trans | - | True、False |
| alpha | - | {0, 0.5, 1.0, 2.0} 平均分布 |
| beta | - | {0, 0.5, 1.0, 2.0} 平均分布 |
| 稀疏度 | - | 50%~99.9% 平均分布 |
| compute_type | - | INT32、FLOAT32 |

## 4. 验收交付件

在社区任务 IT 系统中提交验收时，需要提交以下交付件：

| 序号 | 交付件名称 | 交付件要求 |
|---|---|---|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ；<br> 2. 在 cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以 PR 形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md |
| 2 | 自测用例及测试代码 | 1. 需要清晰列出精度测试 case 和性能测试 case；<br> 2. 需包含测试工程和结果；<br> 3. 测试代码中的 readme 文件需要说明测试步骤，保证验收人可以复现测试结果 |
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 （不需要模板中 TBE 样例结果）；<br> 2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 昇腾开源算子仓 fork 的个人代码仓链接、分支、算子目录；需包含算子工程代码、算子 README 文档、覆盖全规格多组 acl 调用测试代码；<br> 2. 需要在个人仓邀请账号 Ascend-CANN 作为开发者，如下图所示 |

![邀请示意](./pics/invite.jpeg)

### 验收结果反馈

验收以提交验收申请时的代码为准，72 小时内反馈验收结果，如代码更新请重新提交验收申请，验收时间同步刷新。

## 5. PR 申请合入

验收通过后，在昇腾算子开源仓提交 PR 申请，申请将开发完成的算子合入路径：https://gitcode.com/cann/ops-sparse/tree/master/src/spmv 。

## 6. 参考资料

1. Ascend C 算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
2. 算子开发接口文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html
3. Ascend C 在线课程：https://www.hiascend.com/developer/courses/detail/1691696509765107713
4. 代码样例：https://gitcode.com/cann/ops-sparse/tree/master/src/spmv （当前实现为 A2 版本）
5. cuSPARSE SpMV 参考实现：https://docs.nvidia.com/cuda/cusparse/#cusparsespmv

## 7. 特别注意事项

1. 开发过程需严格遵循 Ascend C 编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 开发前请务必阅读【社区任务】流程及注意事项：https://gitcode.com/org/cann/discussions/39 。

## 8. 环境获取（无需修改，使用模板原始内容）

1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend 。
   - **【补充说明】填写示例：本人gitcode账号是 yolo，现在参与社区任务"7月社区任务-aclnnSpMV算子开发"，需要申请A2/A3算力进行任务开发。**

     ![环境截图](./pics/zaixiankaifa1.png)
     ![环境截图](./pics/apply.png)

2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

     ![环境截图](./pics/yunkaifa.png)

3. 如需额外环境资源，请联系昇腾CANN小助手。
