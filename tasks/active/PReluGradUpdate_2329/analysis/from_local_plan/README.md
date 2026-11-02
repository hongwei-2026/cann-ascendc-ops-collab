# PReluGradUpdate problem 2329：设计阶段

本仓库用于提交题包 `PReluGradUpdate_problem_2329_template.zip` 的设计方案。当前处于设计阶段，尚未实现算子或完成 NPU 测试。

- 设计文档（定稿）：[docs/design_final.md](docs/design_final.md)，与 [docs/design.md](docs/design.md) 同步
- 依据题包：`code/samples/reference.py`、`code/samples/case.json`、`code/op_host/`、`code/op_kernel/` 和 `code/tests/ut/`
- 题包目标：Ascend910B；工程 README 提及 CANN 9.0.0/9.1.0，实际构建环境以评测机为准。

## 本轮设计相对初稿的关键修正

1. **分支实现必须用 `Compare + Select`，不能用掩码乘法**。题包 golden 是 numpy `np.where`，与内置 TBE 的 `one_zero + zero_one*weights` 在 Inf 输入上不同：
   - `x<0, g=+Inf`：掩码乘法得 `NaN`，golden 得 `-Inf`
   - `x>0, w=+Inf`：TBE 得 `1 + 0*Inf = NaN`，golden 得 `g`
   两处均已在 numpy 上实测确认。
2. **模板 tiling 把 fp16 与 bf16 都映射到 `half` 实例，是语义 bug**（bf16 与 half 是不同 16 位格式），必须拆成 3 个 tiling key。
3. **fp16/bf16 走 fp32 中间精度是位安全的**：单乘法无累加，20 万组随机输入验证 fp32+RNE 回写与原生低精度乘法位模式完全一致。
4. **权重展开外提**是主要优化：模式 R 下整个 kernel 只构造一次 `wVec`，per-tile 开销归零。

## 注意

GitCode 社区目录 `02-15-PreluGradUpdate` 的现有模板描述的是两个输入、一个输出 `weight_grad`；本题包是三个输入、两个输出 `dx/update`。两者接口不一致，不能把本设计直接提交为该社区任务的实现方案。
