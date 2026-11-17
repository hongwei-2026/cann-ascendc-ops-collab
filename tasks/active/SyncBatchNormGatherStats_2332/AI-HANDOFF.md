# AI 交接：SyncBatchNormGatherStats_2332

> 给后续 AI / 队友：背景、坑、已做、下一步。读完再改代码。  
> 座位：**只写** `people/yuhongming-2026/SyncBatchNormGatherStats_2332/`；台账写本题 `tasks/active/SyncBatchNormGatherStats_2332/`。

---

## 1. 题是什么

算子：`SyncBatchNormGatherStats`（AscendC，910B）。

汇总多卡/多设备的 `total_sum` / `total_square_sum` / `sample_count`，结合 running `mean`/`variance` 与 attr `momentum`/`eps`，输出：

```text
batch_mean, batch_invstd, new_mean, new_variance
```

权威语义：`workdir/code/samples/reference.py`（与 910B 内置实测对齐）。

### 精度铁律（勿破）

1. 浮点输入先 **fp16 量化**再算（fp32 用例也会被 Cast 进同一套 fp16 核语义）。
2. 中间 **fp32**：batch_mean / batch_var / batch_invstd / EMA。
3. 输出再落到 **fp16 网格**（即便声明 dtype 是 fp32）。
4. `count==1` → `count/(count-1)=inf`；**勿擅自修成有限值**。
5. `new_mean` / `new_variance` 在 OpDef 中恒为 **float**；`batch_*` 与输入同 dtype。

### OpDef dtype（4 组）

| total_sum / sq / mean / var / batch_* | sample_count | new_mean / new_var |
|---------------------------------------|--------------|--------------------|
| float                                 | int32        | float              |
| float                                 | int64        | float              |
| float16                               | int32        | float              |
| float16                               | int64        | float              |

---

## 2. 仓库位置（勿串题）

| 用途 | 路径 |
|------|------|
| 代码 | `people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/` |
| 台账 | `tasks/active/SyncBatchNormGatherStats_2332/` |
| 分析 | 本题 `analysis/` |
| 云测 | 本题 `CLOUD-RUN.md` |
| 来源 ZIP | 本机 `D:\华为算子比赛\SyncBatchNormGatherStats_2332.zip`（**不入库**） |

协作仓：`https://github.com/hongwei-2026/cann-ascendc-ops-collab`  
云端常见根：`/mnt/workspace/cann-ascendc-ops-collab`

**禁止串题**：QuantUpdateScatter / ConfusionTransposeD / PRelu 等优化惯性勿套过来。

---

## 3. 基线从哪来

- ZIP 内已是**可编译**功能实现（含完整 Host/Kernel；入库已剔 `build/`）。
- Kernel 已实现：fp16 量化路径、`count==1→+inf`、负 var+eps 补 NaN、按通道多核切分。
- samples 首点极小：`[1,6]` / C=6 / count=100——**只验证正确性**，不能当性能结论。

---

## 4. 已做 / 未做

| 已做 | 未做 |
|------|------|
| 开题脚手架 + ZIP 导入（剔 build） | 用户确认后的深度 Kernel 改动 |
| AI-1/2/3 + **SUMMARY 定稿** | 云 `-e` / 平台 50 点 |
| path-ledger S001 结案 | 性能专题（未开门） |
| STATUS=`分析定稿` | 任务书性能表抄入 materials |

---

## 5. 下一闸

1. 用户确认「可以编码」  
2. 云 `build.sh -j8 -e` 冒烟 + 全文 log 归档  
3. 按 SUMMARY P0：对照 reference / 边界（count==1、fp16 量化 Test_050 类）  
4. 仅门禁未达标再开有界性能（写 path-ledger 新行）
