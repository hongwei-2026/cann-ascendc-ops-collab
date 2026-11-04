# AI 会话卡片 — CtcLossV2_2325（2026-11-04）

| 字段 | 内容 |
|------|------|
| 题目 | CtcLossV2_2325 |
| 角色 | Implementer（Cursor Composer / Auto） |
| 座位 | `people/yuhongming-2026/CtcLossV2_2325/` |
| 用户目标 | 按模板 zip 实现、本地测通、推 GitHub，并落盘本会话 AI 对话记录 |
| 模板 | `/workspace/CtcLossV2_problem_2325_template.zip` |
| 结果 | EXP_C001 本地 `-e` **PASSED**；代码已 push |

---

## 对话时间线（解决本题相关）

### 1. 用户下发题目

- 附图：CANNJudge / ICT 决赛 **CtcLossV2** 题面 + 工程目录（`op_host` / `op_kernel` / `samples` / `tests`）
- 要求：参考 `@CtcLossV2_problem_2325_template.zip`，写入 `@cann-ascendc-ops-collab`，本地测试通过后推 GitHub；**把解决这个问题的所有 AI 对话记录写进仓库并一起推送**

### 2. AI 启动检查与材料解析

- 解压模板，确认与图目录一致：
  - Host：`ctc_loss_v2_def.cpp` / `_infershape.cpp` / `_tiling.cpp`
  - Kernel：`ctc_loss_v2.cpp` / `.h` / `_tiling_data.h` / `_tiling_key.h`
- 阅读 `samples/reference.py`：明确 910B 对齐 **ops_legacy TIK** 语义（窗口 start/end、`count_trace`、`logadd3`、`loss_from`），与 CANN arch35 标准 CTC **不同**
- 样例 shape：`log_probs[10,1,5]`、`targets[1,8]`、`input_lengths=[10]`、`target_lengths=[8]` → `loss[1]`、`log_alpha[1,10,17]`

### 3. 脚手架

- 台账：`tasks/active/CtcLossV2_2325/`（TASK / STATUS / path-ledger / dev-log）
- 座位：`people/yuhongming-2026/CtcLossV2_2325/workdir/code/`（自 zip 展开）
- 更新 `tasks/README.md` Active 索引

### 4. 实现路径（P001）

| 组件 | 做法 |
|------|------|
| InferShape | `neg_log_likelihood=[N]`；`log_alpha=[N,T,2*S+1]`（S=targets 末维） |
| Tiling | 写入 T/N/C/S/blank/zeroInfinity/targetsIsInt64/usedCores；key=mode1（float） |
| Kernel | 移植 reference.py：count_trace → 时序递推 → DMA 写出 loss/alpha |
| 编译踩坑 | 初版 half 实例化与 float LocalTensor DataCopyPad 类型冲突 → 入口统一 float |

### 5. 本地验证

```
source cann-9.1.0/set_env.sh
bash build.sh -e
# PASS: neg_log_likelihood.bin, mismatched=0/1
# PASS: log_alpha.bin, mismatched=0/170
# PASSED
```

硬件：Ascend 910B4 · CANN 9.1.0

### 6. 关联前序会话（同仓协作上下文）

本仓同日还完成过 **ConfusionTransposeD_2323**（用户另开任务）：

1. 用户要求推送 collab 仓 → SSH 公钥配置 → `main` push  
2. 用户附图 ConfusionTransposeD → 初搜仓内无题 → 用户提供 template zip → 实现并 `-e` PASS → push `ea39edd`  
3. hicann 开源参考下载任务中止（网络），未依赖该源码  

上述记录见：

- `tasks/active/ConfusionTransposeD_2323/explorations/EXP_C001-local-910b4-pass.md`
- 本文件仅完整展开 **CtcLossV2** 对话；ConfusionTransposeD 细节以该题 EXP/台账为准

### 7. 废弃 / 未采纳

- 直接复用 arch35 `ctc_loss_v2_fp32.h` 标准 CTC：**否**（与 reference/TIK golden 不一致）
- half schMode 独立实现：样例未覆盖，暂与 float 同路径以满足模板 key 链接

### 8. 交付物路径（图目录对应）

```
people/yuhongming-2026/CtcLossV2_2325/workdir/code/
├── op_host/ctc_loss_v2_def.cpp
├── op_host/ctc_loss_v2_infershape.cpp
├── op_host/ctc_loss_v2_tiling.cpp
├── op_kernel/ctc_loss_v2.cpp
├── op_kernel/ctc_loss_v2.h
├── op_kernel/ctc_loss_v2_tiling_data.h
├── op_kernel/ctc_loss_v2_tiling_key.h
├── samples/  tests/  examples/  build.sh  CMakeLists.txt  README.md
```

### 9. 下一闸建议

- 扩 N>1 / `transpose` 无关的泛化对拍；超 UB 大 S 分块
- 云上 CodeArts / CANNJudge 复测
