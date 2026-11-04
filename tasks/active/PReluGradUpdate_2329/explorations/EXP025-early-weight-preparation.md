# EXP025 — P105 提前准备标量权重

- 日期：2026-11-04；席位：hongwei-2026；状态：本地候选，云测和平台待测。
- 基线：P103 `47db93e`；P104 不采纳，见 DEC002。
- 用户说明没有 profiler，按单项假设继续试验；不将 profile 作为前置阻塞。

## 假设与改动

Process 先调用 CopyIn 再调用 Compute。P103 在输入 DeQue 之后才准备标量权重；P105 将同一权重索引、GetValue 和类型解码块原封不动地移至 Compute 入口、DeQue 之前，尝试利用输入搬运期间的时间。编译器可能已经做同类调度，源代码前移不保证设备收益。

仅内核头文件存在生产代码变化。Host、tiling、UB 分配、队列、输入输出搬运、half/bf16 转换、数值运算和同步保持 P103；6 个向量操作 / 4 道显式 PIPE_V 均未减少。不叠加双缓冲、DataCopy 分支或常驻零缓冲。

## 本地验证

- `python -B tests/weight_schedule_check.py`：对照 Git `47db93e`，验证候选全文只发生原权重块搬移，并检查块内无输入 UB 依赖及 CopyIn/Compute 顺序。
- `python -B tests/compute_schedule_check.py`：原计算语义及显式依赖检查通过。
- `python -B tests/local_semantic_check.py`：原语义和 tiling 检查通过。
- Kernel UTF-8/LF SHA256：`4f100a73615ba10f0416f3cccce8b58eb474b101e01eaafde608f8ce9363520c`。

本地检查不包含 CANN 编译、设备精度或设备计时，不能据此声称平台 PASS 或提速。

## 下一闸

1. 将本轮实时源码送入既有云测流程，SOURCE 中记录 Kernel LF 指纹，核验实际构建副本一致。
2. 三个 dtype key 编译及设备精度通过后，再交平台计时；归档完整日志与截图。
3. 尽量同环境重复对照 P103；无明确收益则恢复 P103，不叠加候选。历史 42.89 分不代表 P105 成绩。

仅修改本地指定席位及本题记录，不提交、不推送。入口：`../deliverables/SUBMIT_LINKS_P105.md`。

## Owner 复审（同日）

- **可试**：相对 P103 仅搬移权重 GM 读+解码到 DeQue 前；已剥离 P104；6ops/4barrier 不变；`weight_schedule_check` / 语义 / 调度 PASS；hash `4f100a73…520c`。  
- **预期保守**：编译器可能已重叠；公开小 shape 未必有感。无 ≥2% 同机收益 → 回 P103。  
- **下一闸**：CodeArts 三 key `-e` → 平台 vs P103。
