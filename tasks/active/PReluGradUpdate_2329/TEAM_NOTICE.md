# 队友同步 — 平台 case4 ACL 507035（2026-11-03）

@`yuhongming-2026` @`JadeYara` — 请先读本页再改 Kernel / 交平台。

## 现状

| 测试点 | 结果 | 说明 |
|--------|------|------|
| 1–3 | **Pass** | 约 713μs；最优 ~4μs（性能另议） |
| 4 | **Runtime Error** `ACL error 507035` | `VECTOR_CORE_EXCEPTION` |
| 5 | Skipped | 被 4 阻断 |

云测（CodeArts）：P029 tip `91f4596` 三 tiling key 编过，样例 dx/update **0/192 PASS**。  
平台交三 dtype 包后 **case4 仍 507035** → 不是单纯「没注册 fp16/bf16」。

## 已否定 / 已踩坑

1. **DtypeBridge / `ToFloat` 显式特化**会编进每个 key → 连 fp32 报 `not support bf16 type cast`（EXP015/P027）  
2. **只开 fp32** 时 1–3 Pass、4 也 507035（缺低精度绑核）  
3. **P029 原生 half Compare/Select** 云测过、平台 4 仍炸（EXP017）  
4. **Select 八参 + BinaryRepeatParams** 易平台 Compile Error（P022/EXP009）— 默认用 **count** 重载  

## 当前道路 P031（工作树，待提交）

代码：`people/hongwei-2026/PReluGradUpdate_2329/workdir/code/`

- float：原生 Compare+Select  
- half：`if constexpr` 内 **Cast↔fp32** 再算（禁止原生 half Compare）  
- bf16：**禁止 Cast**；位扩展/RNE；标量写后补 **S_V / S_MTE3**  
- 工作区 16MB；Select dst≠src1；`Compare(+zero)`  
- per-channel weight：未对齐前缀标量写；`Duplicate` 仅从 32B 对齐 UB 地址启动

详情：[EXP018](explorations/EXP018-case4-unaligned-weight-ub.md) · [EXP017](explorations/EXP017-case4-507035-p030.md) · [path-ledger](path-ledger.md) P026–P031

## 请队友注意

1. **不要改** `people/hongwei-2026/`；各写各的 `people/<自己>/`  
2. 共享台账只改本题 `tasks/active/PReluGradUpdate_2329/`  
3. 默认试交 / 平台链接以 Owner **hongwei** 最新 tip 为准；改试交包先在群/本 STATUS 打招呼  
4. 云上 **每次** `build.sh` 成败都要打 `EXP_*_logs.tar.gz` 下载归档（见 `SYNC_CODEARTS.md`）  
5. 若你本地复现 case4：先检查所有向量指令的 LocalTensor 起始地址是否 32B 对齐，尤其是 `tensor[offset]` 形式

## 下一闸

hongwei：CodeArts 对 P031 跑 `-e` → 过则平台重交。
队友：可读 EXP018；有独立假设先写 `path-ledger` 新行再开道路。
