# AI 对话实录 — LayerNormV3_2326

> 按用户要求，把解决本题的 AI 对话写入仓库。下面是可公开复盘的对话记录（不含密钥）。完整 IDE 流水不入库；本文件和会话卡片是正式记录。

## Turn 1 — 用户

克隆 https://github.com/hongwei-2026/cann-ascendc-ops-collab ，按这个仓库的要求解决附图里的题，把解决文件和这次 AI 对话写进仓库并推到 GitHub。

文件写法对照本仓同类题目，并参考本地模板包 `LayerNormV3_problem_2326_template.zip`。改完后列出相对该模板改过的文件，给出可以点开的超链接。

附图题面要点：

- 标签：开源仓算子工程、ict
- 参考昇腾内置 LayerNormV3 的 TBE，用 Ascend C 做功能一致的算子，走完设计、开发、测试
- 核心功能对齐，支持原算子的数据类型和数据格式；比较从二进制改为逻辑值
- 必须做泛化，覆盖合法输入
- 自己设计全场景自验证；测试用泛化数据看功能、精度、性能
- 精度默认：float 1e-4，half 1e-3，int 0
- 答疑：https://gitcode.com/cann/ops-nn/issues

## Turn 2 — AI（认仓库、认座位）

1. 直连 GitHub 时本机代理 `127.0.0.1:7897` 没开，请求会失败。清掉代理后用 HTTP/1.1 浅克隆 `main` 成功。
2. 读 `AGENTS.md`、`CONTRIBUTING.md`、`people/CLAIM.md`。本题写在已认领座位 `people/yuhongming-2026/`，台账写在 `tasks/active/LayerNormV3_2326/`。不改 hongwei、JadeYara 的目录。
3. 对照本仓已落地的 `CtcLossV2_2325`：个人代码在 `workdir/code/`，对话写在 `tasks/active/<题>/explorations/AI-DIALOG-TRANSCRIPT-*.md` 和会话卡片里。
4. 解压模板。kernel、tiling、infershape 都是 TODO；def 已注册 float / float16 / bfloat16、ND，属性是 `begin_norm_axis`、`begin_params_axis`、`epsilon=1e-5`。样例是 x `[4,6,8]`、gamma/beta `[6,8]`，mean/rstd `[4,1,1]`。

## Turn 3 — AI（语义）

模板 `reference.py` 在 gamma 维数不够时往**右边** `expand_dims`。这和 TBE / numpy 尾维广播不一致，样例碰巧 gamma 维数刚好所以看不出来。

采用的语义：

- 在 `[begin_norm_axis, rank)` 上求均值和总体方差（除以 N）
- `rstd = 1/sqrt(var+epsilon)`
- `y = (x-mean)*rstd*gamma+beta`
- gamma/beta 元素个数必须等于 `x[begin_params_axis:]`
- 扁平下标 `g` 使用 `gamma[g % gammaNum]`，用来覆盖「gamma 短于归一化行」和「gamma 比一行更长」两种合法形状
- 负轴先换成非负
- 计算用 float32，再 cast 回 fp16/bf16。题面里的 int 阈值是平台默认句，本算子 def 没有整数输出

## Turn 4 — AI（实现，P001）

| 组件 | 做法 |
|------|------|
| InferShape | y 与 x 同形；mean/rstd 从 `begin_norm_axis` 起置 1 |
| Tiling | 按行分核；6 块 float UB（x/tmp/gamma/beta/work/raw）；epsilon 以 float 位型放进 int64；workspace 16MB；key 0/1/2 = fp16/fp32/bf16 |
| Kernel | 每行两遍：求和得 mean，再求方差得 rstd，然后仿射写 y。列按 tile 搬入，尾部补 0 再归约，除数仍是真实列长。`gammaNum==1` 时用 Duplicate |
| reference / UT | 尾维广播；host UT 期望串改为 `4 48 48 1 0 4 48 64 925353388 `（样例，core≥4）；kernel UT 单核跑 4 行 |
| 自测 | 新增 `tests/local_semantic_check.py` |

`op_host/layer_norm_v3_def.cpp`、examples、`case.json`、`sample.py`、CMake 和 UT 公共桩保持模板原样。

## Turn 5 — AI（自测）

本机没有 Ascend NPU，不能跑 `bash build.sh -e`。跑了语义脚本，输出：

```
PASS sample y(4, 6, 8) mean(4, 1, 1) tile=48
PASS sample-small-tile y(4, 6, 8) mean(4, 1, 1) tile=7
PASS norm2 y(2, 3, 4, 5) mean(2, 3, 1, 1) tile=9
PASS params-after-norm y(2, 3, 4, 5) mean(2, 3, 1, 1) tile=8
PASS params-before-norm y(2, 3, 4, 5) mean(2, 3, 1, 1) tile=11
PASS negative-axis y(2, 3, 4, 5) mean(2, 3, 4, 1) tile=5
PASS whole-tensor y(4, 6, 8) mean(1, 1, 1) tile=13
PASS scalar-affine y(4, 6, 8) mean(4, 1, 1) tile=10
PASS half rtol=1e-3
ALL PASSED
```

STATUS 写成「实现中」，不写成设备 PASS。

## Turn 6 — AI（落盘并推送）

- 代码：`people/yuhongming-2026/LayerNormV3_2326/workdir/code/`
- 原模板：`tasks/active/LayerNormV3_2326/materials/_unzipped/code/`
- 本对话、会话卡片、EXP_L001、TASK / STATUS / path-ledger / dev-log
- 改过的模板文件链接：`tasks/active/LayerNormV3_2326/SUBMIT_LINKS.md`
- 提交身份按 GitHub DCO：`yuhongming-2026 <hongming@isrc.iscas.ac.cn>`，`commit -s`，推 `main`

## 未采纳

- 把模板右侧 `expand_dims` 当成 TBE 语义：会在 gamma 比归一化轴短时广播错
- 直接搬 ops-nn 的 Welford 大 kernel：和本模板的工程骨架不是同一条路，也超出「先做正确实现」
- 在没有 910B 日志时写 `build.sh -e` PASSED
