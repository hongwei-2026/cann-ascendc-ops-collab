# MultilabelMarginLoss（2327）设计说明

> 实施以 [analysis/SUMMARY.md](analysis/SUMMARY.md) 为准；细则优先 [analysis/AI-codex-claude_design_final.md](analysis/AI-codex-claude_design_final.md)。  
> 初版代码：`people/hongwei-2026/MultilabelMarginLoss_2327/workdir/code/`（融合队友 baseline）。

## 需求

基于 Ascend C 实现与 910B 内置 TBE `MultilabelMarginLoss` **功能对齐**的算子：

- 数据类型：`float32` / `float16` / `bfloat16`（`x`/`y`）；`target`/`is_target` 为 `int32`
- 格式：ND
- 属性：`reduction ∈ {none, mean, sum}`（默认 mean）
- 比较：逻辑比较（`pair <= 0` 才丢弃；NaN 保留）
- 精度：CANN Judge 题目默认阈值
- 性能：全核场景 ≥ TBE 95%；小 shape（<10µs 且差 ≤3µs）可用仿真图说明

## 语义（相对 PyTorch 的 910B 差异）

1. **`is_target` 是出现次数**（非 0/1），dtype 恒 `int32`
2. 每行扫描 `target`，遇到**第一个 `<0`** 截断；前缀内合法类标 `0<=t<C` 计数
3. 损失：

```text
losses[i] = 1/C * Σ_{j:cnt[j]==0} Σ_{l:cnt[l]>0} cnt[l] * max(0, 1 + x[j] - x[l])
```

4. 中间计算 **float32**，再写回输出 dtype

## Host

| 模块 | 策略 |
|------|------|
| InferShape | `is_target` 与 `x` 同形；`none` 且 2D → `y=[N]`，否则 `y=[1]` |
| Tiling | 按 batch 行分核；workspace = 16MB + `N*sizeof(float)` 行损失缓冲 |
| Key | 0=fp16 / 1=fp32 / 2=bf16 |

## Kernel

- 每核处理连续行区间 `[start, start+count)`
- 行内标量实现计数 + O(C²) hinge（功能优先；大 C 后续可向量化）
- fp16/bf16：位转换到 float32 计算；bf16 **不用** `Cast`（910B 缺指令）
- `reduction!=none`：行损失写入 workspace，`SyncAll` 后核 0 做 sum/mean 写 `y[0]`

## 自测

```bash
python3 code/tests/local_semantic_check.py
# 有 NPU / CANN 时：
source /path/to/cann/set_env.sh
cd code && bash build.sh -j4 -e
```

## 下一闸

1. CodeArts / 本地 NPU：`-e` 对齐样例 + 泛化 case  
2. 大 C（如 4096）向量化与性能 profiling  
3. 验收包与合入 `ops-nn` 流程（见社区任务书）
