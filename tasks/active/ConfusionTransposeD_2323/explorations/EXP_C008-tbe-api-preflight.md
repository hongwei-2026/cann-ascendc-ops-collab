# EXP_C008 — 同机 TBE 基线与 910B API 前置盘点

2026-11-11；hongwei-2026；ConfusionTransposeD_2323；P008。

## 假设与预算

P004/P006/P007 的对照均是本算子版本之间的同机 A/B，不能证明相对任务要求 TBE 的实际差距；EXP_C003 的平台 best time 也没有 TBE 身份、输入 shape 或设备 profile 证据。假设 CodeArts 的 CANN 9.1 环境包含可单独识别的 TBE `ConfusionTransposeD` / `npu_confusion_transpose` 实现，并能确认可用 `Transpose`/`Gather` 原语。盘点预算 60 分钟，回传分析预算 30 分钟；这一步不改 Kernel。

只有定位到无自定义算子遮蔽的 TBE 调用方式、输入构造和可识别的 device kernel task 时间，才允许新增路线测量同机 P004/TBE。平台隐藏输入无法映射、TBE 无法独立调用或只取得 Host wall time，都算假设未证实，触止损并 DEC。

## 操作与证据

本机 Windows 环境没有 CANN toolkit；由 Owner 在 910B / CANN 9.1 CodeArts 运行 [`C008-CLOUD-PREFLIGHT.md`](../C008-CLOUD-PREFLIGHT.md) 中的一键盘点。脚本记录环境、TBE 文件与注册符号、Transpose/Gather 头文件及声明，并生成完整日志 tar。当前等待 CodeArts 回传；不能把目录扫描结果写成 TBE 性能结论。

本地检查：脚本通过 Git Bash `bash -n`；恢复基线上的`local_semantic_check.py`全项PASS。尚未运行CANN build或NPU回归，因为本闸没有Kernel改动且本机没有CANN。

## 结论

CodeArts盘点成功（910B3 / CANN 9.1.0，exit=0）。2026-11-12 签名回传进一步确认：

1. **公开 API 属性对齐**：`aclnnConfusionTransposeGetWorkspaceSize(x, perm, shape, transpose_first, out, …)` —— 与题目 attrs 一致。注意 API 名是 `ConfusionTranspose`（无 `D`），竞赛算子仍是 `ConfusionTransposeD`。
2. **TBE→AscendC 分派**：`register_operator("ConfusionTransposeD")`；device 按 tiling key 进六类 `Transpose*`（`arch35/` 模板路径）。这是借鉴线索，**不等于**可直接 `#include` 进自定义包，也不等于 910B 与 arch35 路径已实测同机。
3. **仍缺**：清空 `ASCEND_CUSTOM_OPP_PATH` 后调用系统 opapi 的同输入 device profile；平台隐藏点 shape。

归档：[`C008-api-signature-paste.log`](logs/EXP_C008/C008-api-signature-paste.log)；[`C008-system-symbol-paste.log`](logs/EXP_C008/C008-system-symbol-paste.log)；[`C008-torch-npu-system-call-paste.log`](logs/EXP_C008/C008-torch-npu-system-call-paste.log)。

2026-11-12 可调用闸：`torch_npu.npu_confusion_transpose` + CUSTOM unset + sample 默认 shape，输出 `(2,4,3,5)`，Host 均时 **0.0953 ms/call**（含 launch，非 device task）。**仍未**建立与 P004 同输入的公平对照，也未覆盖平台隐藏点。未完成对照前：**无新 Kernel、不粘贴平台「新优化版」**。
