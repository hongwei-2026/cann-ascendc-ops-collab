# 道路台账 — SyncBatchNormGatherStats_2332

> 规范见 `docs/collaboration/01-path-ledger.md`。开新路前先搜索本表。

| ID | 日期 | 作者 | 阶段 | 假设 | 方法 | 时限 | 结果 | 结论 | 耗时 |
|----|------|------|------|------|------|------|------|------|------|
| S001 | 2026-11-17 | yuhongming-2026 | 开题 | ZIP 基线可编译且已对齐 910B fp16 语义 | 剔 build 导入座位 workdir；建台账；≥3 路分析 + SUMMARY | 1 会话 | 脚手架完成；SUMMARY 定稿 | **采纳（开题）**；深度实现待用户确认 | ~1h |
| S002 | 2026-11-17 | yuhongming-2026 | 功能 | 基线正确性优先：fp16 量化 + 中间 fp32 + count==1→inf | 云 tip `219e221`：`build.sh -j8 -e`；log→`explorations/logs/EXP_S002` | 1 天 | **PASS** 四输出 0/6；编译+样例绿 | **采纳（样例冒烟）**；≠平台 50 点 | ~0.5h |
| S003 | 2026-11-17 | yuhongming-2026 | 精度 | 平台 WA 来自隐藏点（1–5 已 Pass） | 见 HANDOFF-PLATFORM-WA-PROMPT；优先查 count 求和/W>1/fp16 NaN store/count==1 | 1–2 天 | 平台 Wrong Answer；1–5 Pass 0% | **进行中**（交其他 AI） | |
