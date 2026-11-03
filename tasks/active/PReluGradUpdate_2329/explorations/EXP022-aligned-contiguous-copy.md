# EXP022 — P102 对齐 tile 使用连续 DataCopy

| 项 | 记录 |
|----|------|
| 日期 | 2026-11-03 |
| 基线 | P100；EXP_P100 三 key 编译成功、样例 0/192 PASS |
| 历史平台截图 | 2026-11-03 14:46:56，1–5 Pass，42.41 分；截图未显示源码 hash；此前将行内作者标识误记为提交号，现更正 |
| 本实现耗时 | 7.26 / 7.41 / 7.33 / 5.72 / 5.82 μs |
| TBest 耗时 | 4.13 / 4.16 / 4.23 / 3.99 / 4.01 μs |
| 时限 | ≤2 小时；无可重复收益则停止并按 `02-perf-gate` 写 DEC |

## 假设与范围

P100 的 `CopyIn` 和 `CopyOut` 对所有 tile 使用 `DataCopyPad`。设计文档规定对齐搬运优先走普通 `DataCopy`；因此只验证这一项，不同时改变 tiling、dtype 算法、同步或队列数。小 shape 固定开销大，不能仅凭此改动预言平台提分。

只有 `globalOffset * sizeof(T)` 与 `currentNum * sizeof(T)` 均为 32B 整数倍，才选连续 `DataCopy`。四路 GM↔UB 采用相同判定；通道边界、末尾和非对齐 tile 仍由 `DataCopyPad` 处理。此条件比 GM 地址仅按元素对齐更保守，也避免不同核的非对齐输出块相邻写回。

## 验证门禁

1. 本地：`python people/hongwei-2026/PReluGradUpdate_2329/workdir/code/tests/local_semantic_check.py` 与 `git diff --check`。
2. CodeArts：核对源码 SHA256，三 tiling key 编译成功，`build.sh -j4 -e` 样例 0/192 PASS；归档完整日志。
3. 平台：确认提交使用同一源码；对比 1–5 每点耗时，并另测大 shape。只以同机、同版本、重复测量判断收益。
4. 若增益 <2% 或出现错误，回退 P102 候选并写 DEC；保留 P100 可交基线。

## 本地结果

`python -B people/hongwei-2026/PReluGradUpdate_2329/workdir/code/tests/local_semantic_check.py`：PASS；`git diff --check`：PASS。候选 Kernel SHA256：`9E62D7E3C30C874F5E7DC355CDD354D959417DAF6D2C93FDB4EA3CFA61B6F3A0`。

## 云测（EXP_P102_20261103_154833）

- tip `2e3cbc6`；verify PASS；三 key Done；dx/update **0/192 PASS**  
- kernel LF sha256 `0e58c2b2…7221c`（与 git blob 一致）

## Owner 复审（同日）

- **可试且功能已过云测**：单点假设清晰；未动 tiling/Select/dtype。  
- **注意**：公开点 ~7μs vs TBest ~4μs → **B 类**；平台无 ≥2% 增益即回退写 DEC。  
- **下一闸**：同机平台交 P102 采时。

## 平台反馈与撤回（2026-11-03）

用户新图的时间为 **15:48:12**，1–5 仍 Pass，排名 87→95，得分 42.41→42.06。耗时对比如下（旧→新）：

- 点 1：7.26→8.89 μs，增加 1.63 μs（22.45%）。
- 点 2：7.41→8.75 μs，增加 1.34 μs（18.08%）。
- 点 3：7.33→8.99 μs，增加 1.66 μs（22.65%）。
- 点 4：5.72→6.11 μs，增加 0.39 μs（6.82%）。
- 点 5：5.82→6.01 μs，增加 0.19 μs（3.26%）。

截图归档：`screenshots/EXP022-platform-20261103-154812.png`。五点均未见收益，按本实验预先约定，**撤回 P102，恢复 P100**。Kernel 与本地语义检查回到 `3bd65fb` 的内容，保留云测日志及所有历史结论。

### 归因边界

截图没有源码或产物指纹；完整云包 SOURCE 时间 15:48:33 晚于平台图的 15:48:12，不能用该包时间直接证明平台提交来源。较早的 15:44:12 包含 P102 指纹，但日志不完整。需要锁定产物做同环境重复 A/B，区分代码、构建及运行波动。

对齐分支和搬运参数设置可能增加小 tensor 的标量调度成本，这是待 profile 验证的假设，不能据此认定普通 DataCopy 本身更慢。本地语义检查只验证语义和布局模型，不能证明真机性能。

决策见 `../decisions/DEC001-p102-rollback.md`。恢复 P100 后的平台耗时尚未复测，不承诺恢复排名。

### 本地原始日志复核

已核对 `D:\华为算子比赛\EXP_P102_20261103_154833_logs.tar.gz` 及归档目录中的 `SOURCE.txt`、`build_e_full.log`：source/build Kernel 指纹一致，verify PASS，三个 key Done，dx/update 均为 0/192 不匹配。P100 完整构建日志也有三个 key Done 和相同样例 PASS。

原始 P102 包中的 `op_kernel_logs/` 为空目录，没有逐点性能数据、profiler 输出、指令流水线或算子耗时统计。较早 154412 包缺完整构建日志，不能补足这些证据。构建日志里的生成文件名相同也不等于二进制内容相同，不能由此判定错源。

撤回后本地语义检查 PASS；Kernel 与语义检查的规范化 Git 内容分别精确匹配 `3bd65fb` 的 `d8621fee…` 和 `79c59455…`。未进行新的真机编译或平台计时。
