# P004 会话卡片

- 题名：ConfusionTransposeD_2323；成员：hongwei-2026；阶段：性能中。
- 唯一目标：有时限地用连续后缀 DMA 替代标量搬运，保持已通过的功能/精度。
- 协作仓：`D:\sageAttention2 算子开发任务书`，main，起点 545f8d2；已 pull --ff-only，远端一致。
- 隔离工作副本：`D:\华为算子比赛\ConfusionTransposeD_2323_perf_work`，main；完成后仅提交本题路径并 push，原仓再同步。
- 允许：本成员本题目录与本题 tasks 台账、日志、DEC；禁止其他题、他人目录、全局规范。
- 路线：path-ledger P004；实现/profile 主动用时 ≤90 分钟；首轮云测回传后最多 30 分钟修复；小 shape 总计 ≤2 小时。等待 Owner 执行不冒充实测或消耗 profile 预算。
- 云端：Owner 手动执行 CodeArts 命令并回传完整日志包；此会话无直接云端连接。
- 验收：local_semantic_check；build.sh -e；原 13 项 NPU 回归（缺 bf16 须记 SKIP）；新增 DMA 边界回归；同机 baseline/candidate profile。
