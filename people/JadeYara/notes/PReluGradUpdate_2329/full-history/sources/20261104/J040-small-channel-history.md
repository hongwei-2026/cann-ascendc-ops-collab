# 最近三次实验的小通道计时查询

这是已有数据查询，没有新增设备实验。J035、J038、J039 的完整性能输入列表中，小通道样本只有 **N=600000、C=1、S=1**，三种dtype，权重长度1，总计600000元素。其他性能形状的C均至少6656。因此这些记录不能回答小N、小C、低总元素量的近期延迟；不能把S=1的大通道样本当作小通道。

本表是ordinary msprof的设备 **Task Duration(us)**，不是aiv_time或host耗时。每组250次、窗口101–250取中位数；每轮同版本两个组中位数取均值，再平均四轮。括号范围是这四个轮次统计值的最小/最大，不是全部原始调用范围或置信区间。A/A和B/B是同轮同版本前后两组中位数差的最大绝对值。只在同一次实验的A/B之间直接比较；不同实验之间设备状态和运行顺序会变，不能把三次结果拼成单次A/B。

| 实验 | dtype | A版本/均值μs（四轮范围） | B版本/均值μs（四轮范围） | 最大A/A差μs | 最大B/B差μs |
|---|---|---|---|---:|---:|
| J035 | bf16 | J034 8.805000（8.310000–9.090000） | J035 8.615000（8.270000–9.060000） | 0.860000 | 1.060000 |
| J035 | fp16 | J034 8.664625（8.190000–8.980000） | J035 8.694750（8.149500–9.230000） | 0.880000 | 0.781000 |
| J035 | fp32 | J034 9.173687（8.720000–9.470000） | J035 9.007375（8.704750–9.384750） | 0.740000 | 0.940000 |
| J038 | bf16 | J035 8.727375（8.520000–8.939500） | J038 8.847250（8.369500–9.199500） | 1.319000 | 0.720000 |
| J038 | fp16 | J035 8.728438（8.399500–9.274750） | J038 8.779750（8.260000–9.110000） | 1.340000 | 0.659000 |
| J038 | fp32 | J035 9.162375（8.900000–9.330000） | J038 9.249875（8.830000–9.610000） | 1.220000 | 0.679000 |
| J039 | bf16 | J038 8.988687（8.824750–9.270000） | J039 8.955000（8.750000–9.200000） | 1.340000 | 1.220000 |
| J039 | fp16 | J038 9.044625（8.829500–9.390000） | J039 9.052188（8.799500–9.309500） | 1.639000 | 1.621000 |
| J039 | fp32 | J038 9.369937（9.234750–9.645000） | J039 9.297125（9.099500–9.560000） | 1.210500 | 1.179000 |

三次实验的上述输入及golden哈希已逐项核对相同。该C=1路径不走这些实验修改的split-C分支；各轮差值有方向翻转，不能认定近期小通道加速或退化。

## 精确数据来源

- J035：`E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J035_FP32_UB_LAYOUT_910C_20261105/local-control/J035-analysis.json`（逐轮统计）、`E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J035_FP32_UB_LAYOUT_910C_20261105/local-control/J035-summary.json`（逐组统计）、`E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J035_FP32_UB_LAYOUT_910C_20261105/J035-perf-campaign/bench-inputs.json`（输入及哈希）。
- J038：`E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J038_ZERO_REUSE_910C_20261105/local-control/J038-analysis.json`（逐轮统计）、`E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J038_ZERO_REUSE_910C_20261105/local-control/J038-summary.json`（逐组统计）、`E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J038_ZERO_REUSE_910C_20261105/J038-perf-campaign/bench-inputs.json`（输入及哈希）。
- J039：`E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J039_SELECT_SYNC_910C_20261105/local-control/J039-analysis.json`（逐轮统计）、`E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J039_SELECT_SYNC_910C_20261105/local-control/J039-summary.json`（逐组统计）、`E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J039_SELECT_SYNC_910C_20261105/J039-perf-campaign/bench-inputs.json`（输入及哈希）。
