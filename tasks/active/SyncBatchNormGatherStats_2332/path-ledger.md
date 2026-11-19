# 道路台账 — SyncBatchNormGatherStats_2332

> 规范见 `docs/collaboration/01-path-ledger.md`。开新路前先搜索本表。

| ID | 日期 | 作者 | 阶段 | 假设 | 方法 | 时限 | 结果 | 结论 | 耗时 |
|----|------|------|------|------|------|------|------|------|------|
| S001 | 2026-11-17 | yuhongming-2026 | 开题 | ZIP 基线可编译且已对齐 910B fp16 语义 | 剔 build 导入座位 workdir；建台账；≥3 路分析 + SUMMARY | 1 会话 | 脚手架完成；SUMMARY 定稿 | **采纳（开题）**；深度实现待用户确认 | ~1h |
| S002 | 2026-11-17 | yuhongming-2026 | 功能 | 基线正确性优先：fp16 量化 + 中间 fp32 + count==1→inf | 云 tip `219e221`：`build.sh -j8 -e`；log→`explorations/logs/EXP_S002` | 1 天 | **PASS** 四输出 0/6；编译+样例绿 | **采纳（样例冒烟）**；≠平台 50 点 | ~0.5h |
| S003 | 2026-11-17 | yuhongming-2026 | 精度 | fp16 二次 Cast 吞 NaN/Inf 致隐藏点 WA | 仅改 StoreAsInputDtype：Cast 后标量回写 NaN/±inf；保留 count 求和 | 1–2 天 | 云 `-e` PASS；平台仍 **WA** | **部分采纳**（不够，转入 S004） | |
| S004 | 2026-11-17 | yuhongming-2026 | 精度 | `QuantFp16Vec` 往返吞 `new_variance` NaN/±Inf（`count==1`） | helper 内 half 暂存回写非有限值 + `S_V`/`S_MTE3` | 1 会话 | 云 PASS；平台仍 **WA**；可见点 **~6.05–6.69μs**（劣于 S002） | **废弃**（无 WA 收益 + 性能税） | |
| S005 | 2026-11-19 | yuhongming-2026 | 精度 | 隐藏 WA ≠ 输出 Cast 吞 NaN；疑 count/W>1/dtype | 回退 Quant/Store 至 S002；另路证据（下载工程/构造 case） | 1–2 天 | 回退完成；未取到平台工程证据 | **转 S006**（有限值舍入假说优先） | |
| S006 | 2026-11-19 | yuhongming-2026 | 精度 | `x*(1/n)` vs `x/n` 可越过 fp16 边界致有限值 WA | `accSum`/`accSq` 改向量 `Div`；`tests/s006_rounding.py`；tip `b2271b4` | 1 会话 | 云 `-e`+rounding **PASS** 0/6（EXP_S006）；平台待重交 | **采纳（样例）**；≠平台 50 点 | |
| S007 | 2026-11-19 | yuhongming-2026 | 精度 | 隐藏失败可能涉及 dtype、分块尾部、数值边界、count 或设备维，现有 C=6 冒烟无法覆盖 | 157 例矩阵 + `s007_runner`；单机单 count↔reference；W/count 多义为 PROBE | 1 会话 | 云 **PASS=144 FAIL=1 ERROR=0 PROBE=12**；PROBE 对齐 `world_sum+count_sum`；FAIL=`c524289` batch_mean 大片 0 | **采纳（证据）**；转 S008 修大 C | |
| S008 | 2026-11-19 | yuhongming-2026 | 精度 | 多 tile 时 MTE3 仍在读取输出缓存，下一 tile 的向量操作已复用该缓存；长度和多核也需证据裁决 | 保留 MTE3_MTE2，增加 MTE3_V；原种子重复运行大 C 后回归 157 例 | 1 会话 | 云三次 batch_mean 均从 249859 错降为 0；invstd 仍 7、new_variance 仍 13；完整矩阵严格计数 144 PASS / 1 FAIL / 12 PROBE | **部分采纳**；大面积写零改善，余差转 S009 | |
| S009 | 2026-11-19 | yuhongming-2026 | 精度 | 大 C 余差可能由 fp32/fp64 舍入及设备 sqrt/div 引起；S007 的零超差要求比样例允许比例更严格 | 保留严格结果，附报 samples/sample.py 的每输出超差比例；用原种子复现 fp32 数值模型；待平台或内置裁决 | 1 会话 | 本机模型复现 new_variance 差异 12 处，前 8 个与设备例值一致；设备 invstd 差异未复现 | **进行中**；不擅改核数值语义 | |
| S010 | 2026-11-19 | yuhongming-2026 | 精度 | 大 C 中确定超差的位置若出现在小 C，可能超过样例比例阈值；仍需独立设备证据裁决语义 | 固定原种子抽取 invstd 7 / new_variance 8 个已记录位置，覆盖四组 dtype；本地参考检查、云端重复与内置接口可用性探测 | 1 会话 | 云构建与冒烟通过；4 组各重复 3 次均 invstd 7/15、new_variance 8/15 超差，其余两输出 0/15；平台隐藏工程不可获得 | **进行中**；已复现参考差异，内置及平台语义待裁决 | |

S003/S004 note: 标量保 NaN 族补丁 **禁止重开**。平台 1–5 始终 Pass/0%；隐藏点根因未解。
