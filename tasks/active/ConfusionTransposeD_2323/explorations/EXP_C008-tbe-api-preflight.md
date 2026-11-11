# EXP_C008 — 同机 TBE 基线与 910B API 前置盘点

2026-11-11；hongwei-2026；ConfusionTransposeD_2323；P008。

## 假设与预算

P004/P006/P007 的对照均是本算子版本之间的同机 A/B，不能证明相对任务要求 TBE 的实际差距；EXP_C003 的平台 best time 也没有 TBE 身份、输入 shape 或设备 profile 证据。假设 CodeArts 的 CANN 9.1 环境包含可单独识别的 TBE `ConfusionTransposeD` / `npu_confusion_transpose` 实现，并能确认可用 `Transpose`/`Gather` 原语。盘点预算 60 分钟，回传分析预算 30 分钟；这一步不改 Kernel。

只有定位到无自定义算子遮蔽的 TBE 调用方式、输入构造和可识别的 device kernel task 时间，才允许新增路线测量同机 P004/TBE。平台隐藏输入无法映射、TBE 无法独立调用或只取得 Host wall time，都算假设未证实，触止损并 DEC。

## 操作与证据

本机 Windows 环境没有 CANN toolkit；由 Owner 在 910B / CANN 9.1 CodeArts 运行 [`C008-CLOUD-PREFLIGHT.md`](../C008-CLOUD-PREFLIGHT.md) 中的一键盘点。脚本记录环境、TBE 文件与注册符号、Transpose/Gather 头文件及声明，并生成完整日志 tar。当前等待 CodeArts 回传；不能把目录扫描结果写成 TBE 性能结论。

本地检查：脚本通过 Git Bash `bash -n`；恢复基线上的`local_semantic_check.py`全项PASS。尚未运行CANN build或NPU回归，因为本闸没有Kernel改动且本机没有CANN。

## 结论

CodeArts盘点成功（910B3 / CANN 9.1.0，exit=0），确认存在910B注册的legacy TBE配置、legacy Python实现，以及dynamic TBE/AscendC实现文件。扫描到的`Transpose`/`Gather`声明是ACLNN/ES接口；本次没有证明AscendC Kernel内可直接调用的Transpose/Gather原语。

仍未确认TBE实现如何绕开自定义op调用、怎样用P004相同输入做profile，也未取得平台点shape。原始tar留在CodeArts且未下载；只归档用户粘贴的marker段[`C008-console-evidence.log`](logs/EXP_C008/C008-console-evidence.log)，SHA256为`B12C65694536E0D305DD920A6EE42ED0DC10F41CE861D97C5276D736649A9C03`，不宣称完整日志包已归档。当前P008继续限定在30分钟内读取TBE实现/入口；在同机device task对照可执行前，无Kernel diff、无性能PASS。
