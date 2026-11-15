# EXP_C010 — 超 UB 不拒绝 + 连续行吃满 UB（P010）

2026-11-15；hongwei-2026；ConfusionTransposeD_2323。

## 针对的既有问题

1. **平台隐藏点 Runtime Error**：C005 显示 Host 对「整输入装不下 UB」直接 `ACL561002`；P004 硬拒绝是主嫌疑。
2. **连续行路径 UB 被饿死**：即使 `copySpan>1` 不走整输入常驻，旧 tiling 仍先按「整输入占 UB」算 `ubFactor`，行 DMA 块偏小，平台点易停在 ms 级。
3. **相对 TBE ~4.8μs 仍差两个数量级**：本路不宣称追平 TBE；先消 RE + 放大连续行 DMA 收益。P006/P007 微合并已止损（DEC006/007）。

## 假设与预算

- 去掉超 UB Host 失败，改为 `needFullUb=0` + Kernel 从 GM `GetValue` gather → 隐藏大 tensor 可跑通（正确优先）。
- `copySpan>1` 时 `ubFactor=(UB-reserved)/typeSize` → 同行路径吞吐上升。
- 实现≤2h；本地语义检查必过；云测/平台回传后≤1h 分析；小 shape 回退>5% 或功能回归则 DEC 回退。

## 改动

- `tiling_data.h`：新增 `needFullUb`
- `tiling.cpp`：三分派（行 DMA / UB 常驻 / GM gather），**删除**超 UB `GRAPH_FAILED`
- `confusion_transpose_d.h`：`ProcessGmGather` + `ComputeTileFromGm`

无 `fprintf` / `CTD_PROBE`（平台禁 Debug）。

## 结果

待 CodeArts/平台回传。本地：`local_semantic_check.py` 见 dev-log。
