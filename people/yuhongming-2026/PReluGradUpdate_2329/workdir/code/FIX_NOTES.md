# PReluGradUpdate 修复记录

## 日志定位

- build_e_full.log 显示 Host、Kernel、打包和样例编译全部成功，失败只发生在数值校验。
- dx.bin 与 update.bin 都是 mismatched=95/192。
- 固定随机种子下，features <= 0 恰好也是 95 个元素，因此失败集中在负分支；正分支结果基本正确。
- 原远端实验源码未随日志包保存，本地模板仍是 TODO，占位代码无法复现该实现。因此本次在本地重建了完整、可审查的实现，并用显式分支极性锁定语义。

## 核心修复

- CompareScalar(mask, features, 0, GT) 生成正分支掩码。
- Select 遵循 mask=1 -> src0：dx 使用 src0=grads、src1=weights*grads；update 使用 src0=0、src1=features*grads。
- 权重按 channel=(flat_index / innerSize) % channelNum 展开，避免错误地按末维广播。
- fp16/bf16 统一升到 fp32 计算，再用 RNE 写回。

## 结构与性能

- float/fp16/bf16 使用独立 tiling key，避免把 bf16 当 half 解释。
- 小 shape 单核单 tile，减少启动和搬运开销。
- 大 shape 启用多核；当一个 sample 可放入 UB 时，权重向量每核只构造一次并跨 tile 复用。
- 非 32B 尾块统一使用 DataCopyPad，写回只覆盖真实字节。

## 验证

本地无 CANN/NPU 工具链时先执行：python3 tests/local_semantic_check.py

正式环境执行：bash build.sh -j4 -e

并补跑 bash build.sh -j4 -u（配置好 GoogleTest、nlohmann/json 与 tikicpulib 后）。

## EXP_P017 云编补丁（CANN 9.1）

`SyncFunc<HardEvent::S_V>()` 在 9.1 未声明 → opc 三条 key 全挂。  
已改为 `FetchEventID` + `SetFlag`/`WaitFlag`（见 EXP006）。

## EXP_P019

源码指纹 PASS 后仍挂：`not support bf16 type cast`。  
bf16 禁止 `Cast`，改位扩展/RNE（见 EXP002/EXP007）。

## P022（yuhongming）— Compare/Select 通路

云测 WA 95/97 ≡ mask 全 0/全失效，不是极性反。处置：

- 弃用 `CompareScalar`；`Duplicate(0)` 后 `Compare(mask, x, zero, GT)`。
- `Select` 改显式 `maskBits=64, repeats=cmpCount/64, BinaryRepeatParams{1,1,1,8,8,8}`。
- 保留独立 `neg`/`fg` buffer；bf16 仍位转换。
