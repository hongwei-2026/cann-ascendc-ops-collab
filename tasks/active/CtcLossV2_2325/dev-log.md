# Dev log — CtcLossV2_2325

## 2026-11-04

- 从 template zip 建座位与台账
- Host: InferShape `[N]` / `[N,T,2S+1]`；Tiling 按 batch 分核
- Kernel: count_trace 窗口 + logadd3 + loss_from（对齐 reference.py）
- 本地 910B4 `bash build.sh -e`：**PASSED**（loss 0/1，alpha 0/170）
- 落盘 AI 会话卡片与对话实录摘要
