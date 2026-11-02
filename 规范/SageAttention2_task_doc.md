# sageAttention2 算子开发任务书

## 基础信息

- **技术标签**：算子开发 / 量化注意力
- **适配硬件**：Atlas 950PR
- **开源仓地址（本任务交付）**：https://gitcode.com/cann/ops-transformer （合入目录：`experimental/attention/sageattention2`）
- **性能对照仓（FIA 基线）**：https://gitcode.com/cann/ops-transformer （目录 `attention/fused_infer_attention_score/`）
- **CANN 版本**：CANN 9.2.0-beta.1（性能对标与验收环境）
- **开发框架 / 编程语言**：**AscendC** + https://gitcode.com/cann/catlass （Kernel）+ Python（PyTorch 适配层）
- **对标参考**：https://github.com/thu-ml/SageAttention （**`main` 分支** / PyPI `sageattention==2.2.0`，SageAttention2）
- **性能对照 / 门禁基线**：上述 ops-transformer 仓 **FIA**（`FusedInferAttentionScore`）；对外接口见下文 **aclnn** 与 **PyTorch** API（以验收环境集成版本为准）
- **性能硬性要求**：**相对上述 FIA，同机同 shape，aclnn 接口与 PyTorch 接口两条对比路径均须达到 1.6 倍以上（Speedup ≥ 1.6×）**
- **论文**：SageAttention2 https://arxiv.org/abs/2411.10958 （ICML 2025）
- **本任务范围**：**仅** INT8 QK^T + **FP8 PV**（SageAttention2）；不含 FP16 PV、varlen、triton 等路径

## 任务概述

参考开源 **SageAttention2** 中的 **INT8 QK^T + FP8 PV** 路径（精细粒度量化、两级累加、outlier smoothing），在昇腾 NPU 上基于 **AscendC + CATLASS** 实现 **接口对标、功能对齐** 的量化注意力算子，验收通过后合入 **https://gitcode.com/cann/ops-transformer** 仓的 **`experimental/attention/sageattention2`** 目录。

**验收口径**：以 **PyTorch 层接口** 的功能、精度为验收基准；须覆盖自动分发入口与 FP8 显式 API；**aclnn 接口为必选交付**；精度满足《生态算子开源精度标准》及下文量化 Attention 双层策略；**性能须相对 ops-transformer 仓 FIA，在 aclnn 与 PyTorch 两条接口路径上均达到 1.6 倍以上（≥ 1.6×）**。

### 功能定义

本任务实现路径：

| 特性 | 说明 |
|------|------|
| **QK 量化** | INT8；`qk_quant_gran`: `per_warp` / `per_thread`（默认 `per_thread`）。二者为开源 **算法粒度**（非 AscendC 内置枚举）：AscendC 高阶量化多为 PER_TENSOR/CHANNEL/TOKEN/GROUP，**无**同名 `per_warp`/`per_thread` 开关；须用 AscendC Vector（或 Host+Kernel）按开源 scale 分块语义自研 Quant，将粒度映射到 NPU Tile/Block（允许按 Cube/CATLASS 微调分块，**精度与 GPU 同配置对齐**） |
| **PV 量化** | V 量化为 **FP8**，配合 scale |
| **Thorough outlier smoothing** | `smooth_k`（默认 True）；可选 `smooth_v` |
| **两级累加** | `pv_accum_dtype`: `"fp32"` / `"fp32+fp32"` / `"fp32+fp16"`（默认 `"fp32+fp16"`） |
| **head_dim pad** | &lt;64 pad→64；64&lt;D&lt;128 pad→128；D&gt;128 报错 |
| **return_lse** | 可选返回 exponentiated attention weights 的 log sum / logsumexp（Ring Attention 等；语义对齐开源 `core.py`） |
| **自动后端选择** | `sageattn(...)` 在 NPU 上固定分发到 FP8 PV 实现 |

数学形式：

```text
O = softmax(QK^T · s) V
```

其中 Q、K 经 smoothing 后 INT8 量化，V 量化为 FP8。

### 典型应用场景

视频/图像生成（CogVideoX、HunyuanVideo、Flux 等）、LLM/DiT 推理；可 `F.scaled_dot_product_attention = sageattn` 做 plug-and-play（部分自定义 Attention 需改 processor，见开源 example）。

**本任务明确不交付**：`sageattn_qk_int8_pv_fp16_*`、`sageattn_varlen`、以及任何 FP16 PV 实现。

---

## 核心开发要求及验收标准

请参考本任务给出的[测试用例和指导](./sageAttentionTest/)完成测试。

### 1. 须对齐的公开 API

NPU 交付接口对标开源 `sageattention/core.py` 中的 **FP8 PV** 路径：

| NPU API（本任务交付） | 对标开源 API | 说明 | NPU 必选 |
|----------------------|--------------|------|----------|
| `sageattn` | `sageattn` | 自动选择；NPU 上固定走 FP8 PV | **必选** |
| `sageattn_qk_int8_pv_fp8_asc` | `sageattn_qk_int8_pv_fp8_cuda` | INT8 QK + FP8 PV；参数列表、默认值、返回值语义与对标接口一致 | **必选**（950PR 原生 FP8） |
| aclnn FP8 PV 入口（本算子） | 对齐本任务 PyTorch FP8 API 语义 | 统一 Host 侧 aclnn 封装，供性能对比与产品化调用 | **必选** |

> - **`asc` 为 Ascend/NPU 后端命名**；功能与语义对标 `sageattn_qk_int8_pv_fp8_cuda`，实现须基于 **AscendC + CATLASS**，不得依赖 CUDA。
> - **不要求**实现或导出开源 `*_sm90`、`*_cuda` 同名符号（golden 比对仍使用 GPU 侧 `sageattn_qk_int8_pv_fp8_cuda`）。
> - **aclnn 与 PyTorch 均为必选交付**；性能门禁须在两条接口路径上分别对 FIA 达标（见性能要求）。

### 2. PyTorch 层接口（**必选；参数对标开源**）

对标源码：https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py 中的 `sageattn`、`sageattn_qk_int8_pv_fp8_cuda`（本任务 NPU 侧函数名改为 `*_asc`）。

#### 2.1 函数签名

```python
from typing import Any, Optional, Union, Tuple
import torch

def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: ...

def sageattn_qk_int8_pv_fp8_asc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: ...
```

> `sageattn_qk_int8_pv_fp8_asc` 的参数名、类型、默认值、返回值须与开源 `sageattn_qk_int8_pv_fp8_cuda` **逐项对齐**（仅函数名由 `*_cuda` 改为 `*_asc`）。

#### 2.2 `sageattn` 参数说明

对标开源 `sageattn`：按后端能力自动选择实现。开源按 CUDA arch 分发（sm80→FP16 PV、sm89/sm120→FP8 PV 等）；**本任务 NPU 上固定分发到 `sageattn_qk_int8_pv_fp8_asc`**（见 §2.4）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `q` | `torch.Tensor` | — | Query。`HND`：`[batch_size, num_qo_heads, qo_len, head_dim]`；`NHD`：`[batch_size, qo_len, num_qo_heads, head_dim]` |
| `k` | `torch.Tensor` | — | Key。`HND`：`[batch_size, num_kv_heads, kv_len, head_dim]`；`NHD`：`[batch_size, kv_len, num_kv_heads, head_dim]` |
| `v` | `torch.Tensor` | — | Value。shape 与 `k` 相同约定 |
| `tensor_layout` | `str` | `"HND"` | 张量布局：`"HND"` 或 `"NHD"` |
| `is_causal` | `bool` | `False` | 是否对 attention 施加因果 mask；**仅当 `qo_len == kv_len` 时合法** |
| `sm_scale` | `Optional[float]` | `None` | Softmax 前缩放；`None` 时取 `1.0 / sqrt(head_dim)`（`head_dim` 为 pad 前原始维度） |
| `return_lse` | `bool` | `False` | 是否返回 attention 权重经指数化后的 **log sum**（即 softmax 归一化因子的对数）。用于 Ring Attention 等场景。Default: `False`。（对齐开源 docstring：*Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.*） |
| `**kwargs` | `Any` | — | 开源入口主要用于兼容扩展；**NPU 允许**经 kwargs / 环境变量覆盖下沉路径的 `pv_accum_dtype` 等（须在 README 文档化）。**不支持** `attn_mask`（非 `None` 须报错） |

**返回值**（对齐开源 `Returns`）

| 返回项 | 说明 |
|--------|------|
| `o` : `torch.Tensor` | 输出张量。`HND`：`[batch_size, num_qo_heads, qo_len, head_dim]`；`NHD`：`[batch_size, qo_len, num_qo_heads, head_dim]`（裁剪 pad 后的 `head_dim`；dtype 与 `q` 相同） |
| `lse` : `torch.Tensor` | **仅当 `return_lse=True` 时返回**。矩阵 \(QK^\top \cdot scaling\) 每一行的 **logsumexp**（softmax 归一化因子的对数）。Shape：`[batch_size, num_qo_heads, qo_len]`。（对齐开源：*The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).*） |

当 `return_lse=False` 时仅返回 `o`；为 `True` 时返回 `(o, lse)`。

**约束（对齐开源 Note）**

- `num_qo_heads` 须能被 `num_kv_heads` 整除（GQA）
- `q`/`k`/`v` dtype 为 `torch.float16` 或 `torch.bfloat16`，且三者一致、同 device
- NPU 实现：三者须在同一 NPU device（对标开源「同一 CUDA device」）

#### 2.3 `sageattn_qk_int8_pv_fp8_asc` 参数说明

对标开源 https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py （INT8 Q/K + FP8 PV）。除函数名 `*_asc` 外，参数语义须一致。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `q` | `torch.Tensor` | — | Query。布局与 shape 同 §2.2 |
| `k` | `torch.Tensor` | — | Key。布局与 shape 同 §2.2 |
| `v` | `torch.Tensor` | — | Value。布局与 shape 同 §2.2 |
| `tensor_layout` | `str` | `"HND"` | `"HND"` 或 `"NHD"` |
| `is_causal` | `bool` | `False` | 因果 mask；仅 `qo_len == kv_len` |
| `qk_quant_gran` | `str` | `"per_thread"` | Q/K INT8 量化粒度：`"per_warp"` 或 `"per_thread"`（开源算法语义，非法取值须报错）。AscendC 无同名内置开关，实现须自研分块量化并对齐 GPU 同配置数值 |
| `sm_scale` | `Optional[float]` | `None` | Softmax scale；`None` → `1.0 / sqrt(head_dim_og)` |
| `pv_accum_dtype` | `str` | `"fp32+fp16"` | PV 累加精度（对齐开源 FP8 路径实际取值）：<br>• `"fp32"`：全程 FP32 累加（硬件有效位可能受限）<br>• `"fp32+fp32"`：低精度 MMA 累加，定期归约到 FP32 buffer<br>• `"fp32+fp16"`：低精度 MMA 累加，定期归约到 FP16 buffer（**默认验收配置**） |
| `smooth_k` | `bool` | `True` | 在序列维对 K 减均值（outlier smoothing）；略增开销，多数场景提升精度；GQA 时对 LSE 做 correction（`repeat_interleave` 广播） |
| `smooth_v` | `bool` | `False` | 在序列维对 V 减均值。若 `pv_accum_dtype` 为 `"fp32+fp32"` 或 `"fp32+fp16"`，须 **warning 并忽略**（对齐开源实现） |
| `return_lse` | `bool` | `False` | 是否返回 attention 权重经指数化后的 **log sum**。用于 Ring Attention 等场景。Default: `False`。（对齐开源 docstring：*Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.*） |
| `**kwargs` | `Any` | — | 透传/预留；**不支持** `attn_mask`（非 `None` 须报错） |

**返回值**（对齐开源 `sageattn_qk_int8_pv_fp8_cuda` 的 `Returns`，同 §2.2）

| 返回项 | 说明 |
|--------|------|
| `o` : `torch.Tensor` | 输出张量；布局/shape 约定同 §2.2；dtype 与 `q` 相同 |
| `lse` : `torch.Tensor` | **仅当 `return_lse=True` 时返回**。\(QK^\top \cdot scaling\) 各行的 **logsumexp**（softmax 归一化因子的对数）。Shape：`[batch_size, num_qo_heads, qo_len]` |

**数值实现须对齐开源**（`core.py` 返回路径，非参数 docstring 原文，但验收须一致）：

```python
# 开源 sageattn_qk_int8_pv_fp8_cuda 等价逻辑
if return_lse:
    return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
else:
    return o
```

即：对外返回的 `lse` 须做 `lse / 1.44269504`（\(\log_2 e\) 相关换算）；若 `smooth_k=True`，还须加上 `lse_correction * sm_scale`（GQA 下对 K 均值做 `repeat_interleave` 广播后与 Q 的校正项）。

**约束（对齐开源 Note + 实现断言）**

- `num_qo_heads % num_kv_heads == 0`
- `q`/`k`/`v`：`float16`/`bfloat16`，同 dtype、同 device；末维 `stride(-1)==1`
- `head_dim`：原始 D∈(0,128]；&lt;64 pad→64，(64,128) pad→128；**D&gt;128 报错**；输出裁回 `head_dim_og`
- `qk_quant_gran ∈ {"per_warp","per_thread"}`；`pv_accum_dtype ∈ {"fp32","fp32+fp32","fp32+fp16"}`
- `fp32+fp16` 路径下 V 的 FP8 `scale_max` 等超参对齐开源（如 2.25；其它 accum 多为 448.0）

#### 2.4 `sageattn` 自动分发（NPU 映射规则）

开源按 CUDA arch 在 FP16/FP8 间分发。**本任务 NPU 上仅保留 FP8 PV**：

| 条件 | 默认分发目标 |
|------|----------------|
| 默认（吞吐优先） | `sageattn_qk_int8_pv_fp8_asc`（`pv_accum_dtype="fp32+fp16"`） |
| 可配置覆盖 | 允许 kwargs / 环境变量指定 `pv_accum_dtype` 为 `"fp32"` / `"fp32+fp32"` / `"fp32+fp16"` |

须在 README 文档化分发与覆盖方式。传入非法参数时须明确报错。

### 3. Shape / dtype 约束（汇总）

| 项 | 约束 |
|----|------|
| 输入 dtype | **FP16 / BF16**；与开源 FP8 路径一致 |
| head_dim | 原始 D∈(0,128]；内部 pad 到 64 或 128；**D&gt;128 报错** |
| 末维连续 | `stride(-1)==1` |
| GQA | `Hq % Hk == 0` |
| 不同 Sq/Sk | 非因果支持 |
| 输出 | 与 q 同 dtype；裁剪 pad 维还原 `head_dim_og` |

### 4. 数据类型与实现路径分级（950PR）

| 级别 | 路径 | 硬件映射 | 性能验收 |
|------|------|----------|----------|
| **L1 必选** | INT8 QK + FP8 PV | 基于 **AscendC + CATLASS**：Cube INT8 + **原生 FP8** + 两级累加 | **纳入（aclnn 与 PyTorch 路径 vs FIA 均 ≥1.6×）** |
| **L1 必选** | per_warp / per_thread 量化 | 语义对齐开源；AscendC **不**提供同名内置模式，须自研 Quant + tiling（可参考 PER_GROUP 等作分块基础）；或 Host+Kernel 融合 | **纳入** |

**950PR 能力依据**：原生支持 FP8 / MXFP8 等，单芯片 FP8 约 1 PFLOPS 量级，适合本任务 FP8 PV 路径。

**AscendC + CATLASS 开发要求**：

| 要求项 | 说明 |
|--------|------|
| 编程语言 / 框架 | Kernel **须使用 AscendC 与 CATLASS 联合开发**：CATLASS 负责 Cube/MatMul/FA 类模板与 Block/Tile/Mmad/Layout 等；AscendC 负责 Vector、自定义流水、量化/softmax/epilogue 等不可直接套模板的逻辑。禁止以 CUDA 风格孤立内核或与 AscendC/CATLASS 无关的实现作为主交付 |
| 分工建议 | **CATLASS**：INT8 \(QK^\top\)、FP8 \(PV\) 等 Cube 主循环与可复用 FA tiling；**AscendC**：per-warp/per-thread 量化、smooth、online softmax、两级累加归约、LSE、pad/permute 等 |
| 可参考样例 | CATLASS 中 FlashAttention / MXFP8 FA、低精度 MatMul、Chunk Prefill 等 Ascend950 样例；AscendC 官方 Vector/Cube 编程指南与同类 Attention 算子 |
| 工程集成 | 建议接入 `torch_catlass` / optest / AscendC 工程体系（以仓内现行方式为准）；设计文档须说明 AscendC 与 CATLASS 的模块边界、所用 CATLASS 组件与 Tiling 策略 |
| 量化与 Attn | Quant（INT8 Q/K、FP8 V）与 Attn 主循环优先在 AscendC+CATLASS 内融合或分 stage 流水；Host 侧仅做必要预处理 |

### 5. 算法与 Kernel 要点

1. **基于 AscendC + CATLASS** 组织 Quant + Attn tiling / pipeline（可参考仓内 FA / MXFP8 样例扩展）；
2. **Quant**：复现 per-warp / per-thread INT8 **语义**（AscendC 无同名内置 API；用 Vector 自研或 PER_GROUP 等组合实现，block 可按 Cube/CATLASS Tile 微调，精度须与 GPU 同 `qk_quant_gran` 对齐）；
3. **smooth_k / LSE correction**：与 `core.py` 一致（含 GQA `repeat_interleave`）；
4. **FP8 V**：per-channel（或开源等价）量化 + scale；`fp32+fp16` 时 `scale_max` 等超参对齐开源（如 2.25 vs 448.0）；
5. **两级累加**：低精度 MMA 累加定期归约到 FP32 buffer，抑制误差；
6. **禁止**物化全量 attention 矩阵；
7. **仅前向**验收。

### 6. 接口分层

```
┌──────────────────────────────────────────────┐
│ PyTorch（必选）sageattn / *_pv_fp8_asc       │
├──────────────────────────────────────────────┤
│ aclnn（必选）统一 FP8 PV 入口                │
├──────────────────────────────────────────────┤
│ AscendC + CATLASS Kernel：                   │
│   Quant（INT8，AscendC）+ Attn（FP8 PV，CATLASS/AscendC）│
└──────────────────────────────────────────────┘
```

---

### 功能验收用例（须全部通过）

| 编号 | 场景 |
|------|------|
| TC-01 | `sageattn` 分发至 FP8 PV，FP16/BF16，HND/NHD |
| TC-02 | `sageattn_qk_int8_pv_fp8_asc`：`per_thread`/`per_warp` × `fp32` / `fp32+fp32` / **`fp32+fp16`** |
| TC-03 | `smooth_k` / `smooth_v` 开关与 warning 行为 |
| TC-04 | `return_lse=True`，LSE shape `[B,Hq,Sq]`，数值对齐 GPU 同路径 |
| TC-05 | head_dim pad：D=48→64，D=96→128，输出裁剪正确 |
| TC-06 | GQA + Sq≠Sk |
| TC-07 | causal |
| TC-08 | D&gt;128 / 非连续末维 / 非法 gran / 非空 `attn_mask` 报错 |
| TC-09 | 长序列 S≥8K/16K |
| TC-10 | 与 `F.scaled_dot_product_attention` 替换冒烟（example 级） |

---

### 性能要求

#### FIA 对照算子接口（ops-transformer）

性能基线算子 **FIA**（`FusedInferAttentionScore`）来自：

| 项 | 内容 |
|----|------|
| 代码仓 | https://gitcode.com/cann/ops-transformer |
| 算子目录 | https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score |
| 功能说明 | 适配增量 & 全量推理的 FlashAttention：全量（PromptFlashAttention）/ 增量（IncreFlashAttention） |
| aclnn 文档（950PR 性能对标） | https://gitcode.com/cann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md |
| 算子 README | https://gitcode.com/cann/ops-transformer/blob/master/attention/fused_infer_attention_score/README.md |

> 本任务目标硬件为 **Atlas 950PR**。按当前 ops-transformer / CANN 实现：**Ascend 950PR 须使用 `aclnnFusedInferAttentionScoreV5` 作 aclnn 性能基线**。性能验收对比须基于**最新 CANN 9.2.0-beta.1**；接口以该 CANN / 仓内 **V5** 文档为准。

##### （1）aclnn 接口（950PR：`aclnnFusedInferAttentionScoreV5`）

两段式调用（先申请 workspace，再执行）：

```c
// 第一段：计算 workspace
aclnnStatus aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
    const aclTensor     *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor     *pseShiftOptional,
    const aclTensor     *attenMaskOptional,
    const aclIntArray   *actualSeqLengthsOptional,
    const aclIntArray   *actualSeqLengthsKvOptional,
    const aclTensor     *deqScale1Optional,
    const aclTensor     *quantScale1Optional,
    const aclTensor     *deqScale2Optional,
    const aclTensor     *quantScale2Optional,
    const aclTensor     *quantOffset2Optional,
    const aclTensor     *antiquantScaleOptional,
    const aclTensor     *antiquantOffsetOptional,
    const aclTensor     *blockTableOptional,
    const aclTensor     *queryPaddingSizeOptional,
    const aclTensor     *kvPaddingSizeOptional,
    const aclTensor     *keyAntiquantScaleOptional,
    const aclTensor     *keyAntiquantOffsetOptional,
    const aclTensor     *valueAntiquantScaleOptional,
    const aclTensor     *valueAntiquantOffsetOptional,
    const aclTensor     *keySharedPrefixOptional,
    const aclTensor     *valueSharedPrefixOptional,
    const aclIntArray   *actualSharedPrefixLenOptional,
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    const aclTensor     *keyRopeAntiquantScaleOptional,
    const aclTensor     *dequantScaleQueryOptional,
    const aclTensor     *learnableSinkOptional,
    const aclIntArray   *qStartIdxOptional,
    const aclIntArray   *kvStartIdxOptional,
    int64_t              numHeads,
    double               scaleValue,
    int64_t              preTokens,
    int64_t              nextTokens,
    char                *inputLayout,
    int64_t              numKeyValueHeads,
    int64_t              sparseMode,
    int64_t              innerPrecise,
    int64_t              blockSize,
    int64_t              antiquantMode,
    bool                 softmaxLseFlag,
    int64_t              keyAntiquantMode,
    int64_t              valueAntiquantMode,
    int64_t              queryQuantMode,
    int64_t              pseType,
    const aclTensor     *attentionOut,
    const aclTensor     *softmaxLse,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor);

// 第二段：执行
aclnnStatus aclnnFusedInferAttentionScoreV5(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream);
```

与本任务性能对比相关的**常用参数**（完整参数见仓内 **V5** 文档；相对 V4 新增 `qStartIdxOptional` / `kvStartIdxOptional` / `pseType`）：

| 参数 | 含义（摘要） |
|------|----------------|
| `query` / `key` / `value` | 输入 Q/K/V；dtype 常见 FP16/BF16（及量化场景 INT8 等） |
| `attentionOut` | 输出 Attention 结果 |
| `numHeads` / `numKeyValueHeads` | Query 头数 / KV 头数（GQA） |
| `scaleValue` | Softmax 缩放，通常 `1/sqrt(d)` |
| `inputLayout` | 排布字符串，如 `"BNSD"`、`"BSH"`、`"BSND"` 等（与本任务 HND/NHD 对齐时需做布局换算） |
| `preTokens` / `nextTokens` / `sparseMode` | 稀疏/因果窗口；非因果长序列对比时按 FIA 文档取高性能合法配置 |
| `actualSeqLengthsOptional` / `actualSeqLengthsKvOptional` | 实际序列长度（可选） |
| `softmaxLseFlag` / `softmaxLse` | 是否输出 Softmax LSE |

仓内 README 亦给出 aclnn 调用样例入口（如 `test_aclnn_FusedInferAttentionScoreV5`），验收对比脚本可参考。

##### （2）PyTorch 接口（torch_npu）

FIA 经 Ascend Extension for PyTorch（op-plugin）暴露为 `torch_npu` API。性能验收须分别完成 **PyTorch 层**（本任务 `sageattn_qk_int8_pv_fp8_asc` vs `torch_npu.npu_fused_infer_attention_score`）与 **aclnn 层**（本任务 aclnn 入口 vs **`aclnnFusedInferAttentionScoreV5`**）对比，**两条路径均须 ≥ 1.6×**。

**主接口（常用）**：

```python
import torch
import torch_npu

# 返回 (attention_out, softmax_lse)；不需要 LSE 时可忽略第二返回值
attention_out, softmax_lse = torch_npu.npu_fused_infer_attention_score(
    query,                              # Tensor, FP16/BF16, ND
    key,                                # Tensor, FP16/BF16/(量化场景 INT8 等)
    value,                              # Tensor, 同 key 约束
    *,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    dequant_scale1=None,
    quant_scale1=None,
    dequant_scale2=None,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale=None,
    antiquant_offset=None,
    block_table=None,
    query_padding_size=None,
    kv_padding_size=None,
    key_antiquant_scale=None,
    key_antiquant_offset=None,
    value_antiquant_scale=None,
    value_antiquant_offset=None,
    key_shared_prefix=None,
    value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None,
    key_rope=None,
    key_rope_antiquant_scale=None,
    num_heads=1,
    scale=1.0,                          # 对应 scaleValue，通常 1/sqrt(head_dim)
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="BSH",                 # 如 "BNSD" / "BSH" / "BSND"
    num_key_value_heads=0,              # 0 表示与 num_heads 相同；GQA 时填 KV 头数
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    antiquant_mode=0,
    softmax_lse_flag=False,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
)
```

**对照本任务基准用例的最小调用示例**（布局 `BNSD` ≈ SageAttention `HND`：`[B, N, S, D]`）：

```python
import math
import torch
import torch_npu

B, N, S, D = 1, 32, 4096, 128
q = torch.randn(B, N, S, D, dtype=torch.float16, device="npu")
k = torch.randn(B, N, S, D, dtype=torch.float16, device="npu")
v = torch.randn(B, N, S, D, dtype=torch.float16, device="npu")
scale = 1.0 / math.sqrt(D)

# FIA 基线
out_fia, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    num_heads=N,
    num_key_value_heads=N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535,
)

# 本任务算子（HND 与 BNSD 同形时可直接对比）
# out_sage = sageattn_qk_int8_pv_fp8_asc(q, k, v, tensor_layout="HND", pv_accum_dtype="fp32+fp16")
```

**可选新接口**：在不涉及 system prefix、左 padding、部分量化合并等场景时，环境若提供 `torch_npu.npu_fused_infer_attention_score_v2`，可按其文档使用；**同一验收报告内 FIA 接口版本须固定**，不得混用 v1/v2 导致口径不一致。文档参见 op-plugin：`torch_npu-npu_fused_infer_attention_score(_v2).md`。

| 对比约定 | 说明 |
|----------|------|
| 布局 | 优先 `input_layout="BNSD"` 对齐本任务 `tensor_layout="HND"`；若用 `BSH`/`BSND`，须在报告中说明换算与是否计入转置开销 |
| dtype | 与基准用例一致（FP16/BF16）；FIA 走其默认高精度/推荐配置，**不强制**与本算子同为 INT8+FP8 |
| GQA | `num_key_value_heads` 与本任务 `Hkv` 对齐 |
| 因果 | 按 FIA `sparse_mode` / `pre_tokens`/`next_tokens` 文档配置合法因果模式，与本任务 `is_causal=True` 语义对齐 |

#### 性能门禁（硬性指标）

**本任务最终性能要求：在 Atlas 950PR、CANN 9.2.0-beta.1 同机、同 shape、同 dtype、同等预热与计时口径下，相对 https://gitcode.com/cann/ops-transformer 仓 FIA，须同时满足：**

1. **PyTorch 路径**：本算子 PyTorch API vs `torch_npu.npu_fused_infer_attention_score`，Speedup ≥ 1.6×  
2. **aclnn 路径**：本算子 aclnn API vs `aclnnFusedInferAttentionScoreV5`，Speedup ≥ 1.6×  

任一路经未达标不予验收。

| 项 | 要求 |
|----|------|
| 对照算子（基线） | **ops-transformer 仓 FIA**（`FusedInferAttentionScore`）；**PyTorch**：`torch_npu.npu_fused_infer_attention_score`（或验收环境指定的 `_v2`）；**aclnn（950PR）**：`aclnnFusedInferAttentionScoreV5`。须在自验证报告写明 **CANN 9.2.0-beta.1 + 实际调用的 API 名** |
| 本算子（必选双接口） | **PyTorch**：`sageattn_qk_int8_pv_fp8_asc`（默认 `pv_accum_dtype="fp32+fp16"`）或经 `sageattn` 分发的等价路径；**aclnn**：本任务交付的统一 FP8 PV aclnn 入口（语义与 PyTorch 路径对齐） |
| 达标判据 | **aclnn 与 PyTorch 两条路径各自** Speedup ≥ 1.6×；基准用例 **每一条、每一路径**均须达标；同时分别报告两路径全用例几何平均 Speedup（均须 ≥ 1.6×） |
| 计时口径 | 排除首次编译/JIT；充分 warmup 后取稳定 kernel/算子耗时（PyTorch 推荐 `torch.npu.Event`；aclnn 用等价 Host/Device 计时）；batch 内同步后统计；自验证报告分列两路径计时与参数 |
| 对比公平性 | 同一设备、同一输入 shape（HND/NHD 与 FIA `input_layout` 转换开销若计入须双方一致说明）；FIA 使用 ops-transformer / CANN 文档推荐的高性能配置；若 FIA 不支持某 dtype/布局，该条改用 FIA 最近合法等价配置并在报告中标注 |
| 实现约束 | Kernel **须基于 AscendC + CATLASS**；**PyTorch 与 aclnn 均为必选交付**；验收取该路径实测最优实现 |

> - **性能 golden**：ops-transformer 仓 **FIA**；须同时完成 **aclnn 与 PyTorch** 两套对比；GPU SageAttention 对照数据仅可作分析加分项，**不作为**性能门禁。
> - 「1.6 倍以上」验收口径为 **Speedup ≥ 1.6**（\(T_{\text{本算子}} \le T_{\text{FIA}} / 1.6\)），对两条接口路径分别判定。

#### 分档说明（aclnn 与 PyTorch 路径均须 ≥ 1.6× FIA）

| 分档 | 场景 | 本算子配置 | **达标** |
|------|------|------------|----------|
| **S1** | 典型长序列 | `fp32+fp16` | **aclnn 与 PyTorch 均 ≥ 1.6× FIA** |
| **S2** | 因果 | `fp32+fp16` | **aclnn 与 PyTorch 均 ≥ 1.6× FIA** |
| **S3** | GQA | `fp32+fp16` | **aclnn 与 PyTorch 均 ≥ 1.6× FIA** |
| **S4** | BF16 输入 | `fp32+fp16` | **aclnn 与 PyTorch 均 ≥ 1.6× FIA** |
| **S5** | 更长序列 | `fp32+fp16` | **aclnn 与 PyTorch 均 ≥ 1.6× FIA** |

#### 基准用例（须实测填列；两路径均填）

| 编号 | shape（HND） | 本算子配置 | 分档 | \(T_{\text{FIA-PT}}\) | \(T_{\text{本算子-PT}}\) | Speedup-PT | \(T_{\text{FIA-aclnn}}\) | \(T_{\text{本算子-aclnn}}\) | Speedup-aclnn | 两路径均 ≥1.6× |
|------|--------------|------------|------|----------------------|-------------------------|------------|--------------------------|----------------------------|---------------|----------------|
| P-01 | `[1,32,4096,128]` | fp8_asc，**fp32+fp16** | S1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 门禁 |
| P-02 | `[1,32,8192,128]` | fp8_asc，`fp32+fp16` | S1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 门禁 |
| P-03 | `[1,32,8192,128]` causal | fp8_asc，`fp32+fp16` | S2 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 门禁 |
| P-04 | `[2,32,8192,128]` | fp8_asc，`fp32+fp16` | S1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 门禁 |
| P-05 | GQA Hq/Hk=32/8，S=4096，D=128 | fp8_asc，`fp32+fp16` | S3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 门禁 |
| P-06 | `[1,32,4096,128]` BF16 | sageattn 默认（`fp32+fp16`） | S4 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 门禁 |
| P-07 | `[1,32,16384,128]` | fp8_asc，`fp32+fp16` | S5 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 门禁 |

> 开发者提交验收前，须在自验证报告中完整填写上表（**PyTorch 与 aclnn 两列均不可空**），并附两路径 profiler / 计时日志，同时注明对照 FIA 所用的 **CANN 9.2.0-beta.1**。

---

### 精度要求

须满足《生态算子开源精度标准》https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md ，使用 AscendOpTest。

#### 双层策略

| 层级 | 标杆 | 说明 |
|------|------|------|
| **L-A（主）** | GPU 对标接口 `sageattn_qk_int8_pv_fp8_cuda`（同超参） | NPU 调用 `sageattn_qk_int8_pv_fp8_asc`；混合容差：FP16/BF16；（中间 FP8 态可按标准 FP8 表做单测） |
| **L-B（辅）** | FP32 SDPA | NPU 与 GPU-Sage 相对真值的误差比例：max≤2，mean≤1.2，RMSE≤1.2 |

**混合容差（输出为 FP16/BF16 时）**：

| dtype | rtol | atol | matched_ratio | max_abs_error_limit |
|-------|------|------|---------------|---------------------|
| FP16 | \(2^{-9}\) | \(2^{-9}\) | ≥0.99 | 1e-1 或 32×ULP |
| BF16 | \(2^{-6}\) | \(2^{-6}\) | ≥0.99 | 1e-0 或 32×ULP |

**路径特化说明**：

| 路径 | 精度策略 |
|------|----------|
| FP8 PV + `fp32+fp16` | L-A 对 GPU 同 `pv_accum_dtype`；**不得**用 FP16-PV 或其他 accum 配置当 golden |
| FP8 PV + `fp32` / `fp32+fp32` | 分别对标 GPU 同 `pv_accum_dtype` |
| `return_lse` | LSE 使用 FP32 容差或相对 GPU LSE 的混合容差 |
| per_thread vs per_warp | 分别对标，不可混用 golden |

---

### 交付件要求

| 序号 | 交付件名称 | 交付件要求 |
|------|------------|------------|
| 1 | 算子设计文档 | 1. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md ；<br>2. 在 cann-competitions 仓库（https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist ）以 PR 形式提交设计文档，通过评审后合入仓库，详细说明见：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md |
| 2 | 自测用例及测试代码 | 1. 需要清晰列出精度测试 case 和性能测试 case；<br>2. 测试代码中的 readme 文件需要说明测试步骤，保证验收人可以复现测试结果 |
| 3 | 自测报告 | 1. 自测报告模板：https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2 ；<br>2. 需要包含用例参数、精度对比结果及截图、性能数据及截图 |
| 4 | 待验收代码地址 | 1. 个人代码仓链接、分支、算子目录；需要在个人仓邀请账号 Ascend-CANN 作为开发者；<br>2. 需要根据每个仓库的规范提供算子 readme 文档 |

---

## 验收规则与流程

交付三类材料（代码仓、自验证报告、设计文档）→ 验收 → PR 合入：

```
https://gitcode.com/cann/ops-transformer
```

**合入路径**：

```
https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/sageattention2
```

建议目录结构：

```text
experimental/attention/sageattention2/   # 本任务交付根目录
├── sageattn_fp8/                        # AscendC + CATLASS kernel
├── quant/                               # 量化相关
├── python/                              # PyTorch 适配层（必选）
├── aclnn/                               # aclnn 接口（必选）
├── docs/                                # 设计文档 / 自验证报告（可选）
└── README.md
```

> 交付合入目录为仓内 **`experimental/attention/sageattention2`**；性能对照基线为同仓 **FIA**（`attention/fused_infer_attention_score/`），二者目录与用途不同，勿混淆。

---

## 参考资料

1. https://github.com/thu-ml/SageAttention （对标：`sageattention/core.py` 中 `sageattn_qk_int8_pv_fp8_cuda`、`quant.py`、`sm89_compile`）
2. **交付合入仓**：https://gitcode.com/cann/ops-transformer （目录 `experimental/attention/sageattention2`）
3. **CATLASS**：https://gitcode.com/cann/catlass （含 Ascend950 FA / MXFP8 等样例）
4. **AscendC**：CANN Ascend C 编程指南 / 算子开发文档（以验收环境版本为准）
5. **性能基线 FIA 仓**：https://gitcode.com/cann/ops-transformer  
   - 算子：`attention/fused_infer_attention_score/`  
   - aclnn（950PR）：`aclnnFusedInferAttentionScoreV5`（https://gitcode.com/cann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md ）  
   - PyTorch：`torch_npu.npu_fused_infer_attention_score`（及可选 `_v2`，见 Ascend op-plugin 文档）  
6. 精度标准：opbase `experimental_standard.md`
7. ops-transformer / flash-attention-npu（AscendC / CATLASS 产品化参考；FIA 对照用）
8. 论文 SageAttention2 https://arxiv.org/abs/2411.10958 （arXiv:2411.10958）

---

## 特别注意事项

1. **交付合入**：https://gitcode.com/cann/ops-transformer 的 **`experimental/attention/sageattention2`** 目录；**性能门禁基线**为同仓 **FIA**（`attention/fused_infer_attention_score/`），目录与用途勿混用；
2. **Kernel 须基于 AscendC + CATLASS 开发**，设计文档与代码目录应体现二者分工与 CATLASS 模板复用；不得以纯 Host 或与 AscendC/CATLASS 无关实现作为主交付；
3. **性能门禁（硬性）**：相对 **https://gitcode.com/cann/ops-transformer** 仓 FIA，**PyTorch 路径**（vs `torch_npu.npu_fused_infer_attention_score`）与 **aclnn 路径**（vs **`aclnnFusedInferAttentionScoreV5`**，950PR）**均须 ≥ 1.6×**；P-01～P-07 每条、每路径均须达标，未达标不予验收；**本算子 aclnn 接口为必选实现**；
4. **本任务仅验收 INT8 QK + FP8 PV**；不要求实现 FP16 PV / varlen / triton / `*_sm90` / `*_cuda` 同名路径；
5. **NPU 交付接口名为 `sageattn_qk_int8_pv_fp8_asc`**，参数与语义对标开源 `sageattn_qk_int8_pv_fp8_cuda`；
6. **默认两级累加验收不可缺失**：`sageattn_qk_int8_pv_fp8_asc(..., pv_accum_dtype="fp32+fp16")` 为功能与性能必测项；
7. 精度 golden 必须 **同路径同 accum**（NPU `*_asc` vs GPU `*_cuda`，同 `pv_accum_dtype`）；**性能 golden 为 ops-transformer FIA**，二者勿混淆；
8. `smooth_v` 与 accum dtype 的 ignore/warning 行为须对齐开源；
9. 与 SageAttention1/3 任务在 ops-transformer 仓内分子目录交付，避免包名冲突；
10. 仅前向；分布式 Ring Attention 仅要求 `return_lse` 数值接口正确，不强制多卡用例门禁。

## 环境获取（无需修改，使用模板原始内容）

1. 使用 hidevlab webIDE 算力：https://hidevlab.huawei.com/online-develop-intro?from=hiascend 。
 - **【补充说明】填写示例：本人gitcode账号是 yolo，现在参与社区任务"7月社区任务-aclnnRoll算子开发"，需要申请A2/A3算力进行任务开发。**
 	 
 	![环境截图](./pics/zaixiankaifa1.png)  
 	![环境截图](./pics/apply.png)  

2. 开源仓提供100小时免费时长，请不使用时及时关闭，用时耗尽前请务必保存相关资料，建议及时提交备份。

	![环境截图](./pics/yunkaifa.png)

3. 如需额外环境资源，请联系昇腾CANN小助手。
