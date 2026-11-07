# 对话记录 — PadV3Grad_2328 — 2026-11-07

用户：按前面两题同样的座位、台账、提交和链接流程，参考 `PadV3Grad_problem_2328_template.zip`，把 PadV3Grad 做完整。本机能测的测，没有 NPU 就不测设备。

## 题目

`x` 是 Pad 之后的梯度，`paddings` 说明每一维前面和后面填了多少。输出是原张量上的梯度。

模板 `reference.py` 的规则：

- `paddings_contiguous=true`：`[f0,b0,f1,b1,...]`
- `false`：前 `rank` 个是 front，后 `rank` 个是 back
- constant：填充区没有梯度，取中间
- edge / replicate：下标裁到 `[0, n)`
- reflect：周期 `2(n-1)`，不重复边缘；`n==1` 时全部回到这一个元素
- symmetric：周期 `2n`，重复边缘
- circular：下标对 `n` 取模
- 负数取模与 numpy 相同
- 累加用 float64 再落回输入 dtype。kernel 用 float32 累加，再转回 fp16/bf16

模板 kernel、tiling、InferShape 都是 TODO。样例是 reflect，`x [1,1,8,8]`，paddings `[[0,0],[0,0],[2,2],[2,2]]`，`y [1,1,4,4]`。

## 这次写了什么

- InferShape 用 paddings 把每一维减去 front 和 back。
- constant 按输出的最后一维分行，多核拷贝，workspace 为 0。
- 其它模式单核逐维折回。workspace 是两块 float 缓冲，不调用 SyncAll，所以没有 16MB 系统前缀。
- 注册 fp16/fp32/bf16，以及和 int32、int64 paddings 的组合。模板原来的 def 把 float 和 int32 配不到一起，样例却是这个组合。

## 本机测试

`tests/local_semantic_check.py` 用同一套下标在 float64 里折回，对照 `reference.impl`：

- 五种 mode，外加 replicate
- 不对称 pad，以及前后两段的排布
- pad 大于输出长度时的周期折回
- 输出长度为 1
- fp16、bf16

输出：`PadV3Grad local semantic checks: PASS`。

本机没有 CANN / NPU，没有跑 `build.sh -e`，也没有编译 Host/Kernel UT。不把这次写成设备 PASS。
