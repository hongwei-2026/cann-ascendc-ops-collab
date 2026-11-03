# EXP022 — P102 对齐 tile 使用连续 DataCopy

| 项 | 记录 |
|----|------|
| 日期 | 2026-11-03 |
| 基线 | P100；EXP_P100 三 key 编译成功、样例 0/192 PASS |
| 平台截图 | 提交 13557230920，1–5 Pass，42.41 分；截图未显示源码 hash |
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

本地没有完成 CANN 编译或 NPU 计时；云测和平台结果未取得，P102 不能标记为通过。
