# 道路台账 — MultilabelMarginLoss_2327

| ID | 日期 | 作者 | 阶段 | 假设 | 方法 | 时限 | 结果 | 结论 | 耗时 |
|----|------|------|------|------|------|------|------|------|------|
| M001 | 2026-11-03 | yuhongming-2026 | 准备 | 题包含 reference 与骨架即可开写 | 解压 template；读 reference 与 ops-nn 先例 | 0.5h | 语义差异已确认 | 采纳 | — |
| M002 | 2026-11-03 | yuhongming-2026 | 实现 | 行分核 + 行内标量 O(C²) 可过功能/精度样例 | InferShape/Tiling/Kernel；fp16/fp32/bf16 key；workspace 行损失归约 | 本迭代 | 本地语义 PASS；待云测 | 采纳为 baseline | design.md |
| M003 | 2026-11-03 | hongwei-2026 | 实现 | 融合 teammate + design_final classTile；云端题目隔离 | 拷座位；Host/Kernel 滑窗；写 SYNC_CODEARTS | 本会话 | 本地 semantic PASS；初版落盘 | 采纳；进云测 | FIX_NOTES |
| M004 | 2026-11-03 | hongwei-2026 | 云测 | 首点 -e 可过 | CodeArts build.sh -e | 本会话 | build OK；is_target PASS；**y WA 1/1** | **进行中**：查 hinge/mean 写回；见 EXP_M001 | EXP_M001 |
| 2026-11-03 17:21 | EXP_M001_FIX | Claude/Codex: DataCopyPad StoreY/RowLoss + idle SyncAll + usedCores; local semantic PASS; push 3ebda45 | ������ -e ���� y |
