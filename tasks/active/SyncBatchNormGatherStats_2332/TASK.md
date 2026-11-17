# 任务书摘要 — SyncBatchNormGatherStats_2332

| 项 | 内容 |
|----|------|
| 算子 | SyncBatchNormGatherStats |
| 硬件 | Ascend 910B（OpDef: ascend910b） |
| 输入 | total_sum, total_square_sum, sample_count, mean, variance |
| 属性 | momentum（默认 0.1）, eps（默认 1e-5） |
| 输出 | batch_mean, batch_invstd, new_mean, new_variance |
| 权威语义 | `people/.../workdir/code/samples/reference.py` |
| 座位代码 | `people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/` |
| 来源 ZIP | 本机；**不入库原包** |

## 澄清项（开题已见）

1. fp32 用例也走 fp16 核语义（输入/输出落 fp16 网格）。
2. `new_mean`/`new_variance` OpDef 恒 float；值仍在 fp16 网格。
3. `count==1` 时无偏系数 inf，与设备一致。
4. samples 仅首点；平台约 50 case（以评测为准）。
