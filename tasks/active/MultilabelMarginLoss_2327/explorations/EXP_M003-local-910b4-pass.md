# EXP_M003 — 本地 910B4 同步 tip 后样例 -e PASS

| 字段 | 内容 |
|------|------|
| 题目 | MultilabelMarginLoss_2327 |
| 日期 | 2026-11-04 |
| 作者 / AI | yuhongming-2026 · Cursor Composer（Implementer） |
| 关联台账 | M005 |
| 阶段 | 功能 / 精度（样例） |
| 时限 | 本会话 |
| 结论 | **采纳** |
| tip | 本地 `6f48518`（相对公开 tip `22ec6d6`）；座位同步 hongwei DMA + 去 ScanPrefix const |
| 日志 | [`logs/EXP_M003_20261104_local_e/`](logs/EXP_M003_20261104_local_e/) |
| 会话卡片 | [`AI-SESSION-20261104-card.md`](AI-SESSION-20261104-card.md) |

## 假设

hongwei tip（`3ebda45` DMA 写回 + `22ec6d6` 去掉 `ScanPrefix` const）已修掉 EXP_M001 的 y WA 根因与 EXP_M002 编译失败；把同一 tip **复制进** `people/yuhongming-2026/.../workdir/code/` 后，本机 910B4 应能过首点 `-e`。

依据：[`FIX_NOTES.md`](../../../people/hongwei-2026/MultilabelMarginLoss_2327/workdir/code/FIX_NOTES.md) · EXP_M001/M002 ANALYSIS。

## 做了什么

- **改动路径**（仅本成员座位 + 本题台账）：
  - `people/yuhongming-2026/MultilabelMarginLoss_2327/workdir/code/` ← rsync hongwei tip（Host tiling `classTile` / Kernel DMA StoreY·RowLoss / 非 const `ScanPrefix`）
  - `people/yuhongming-2026/MultilabelMarginLoss_2327/workdir/code/SEAT_NOTES.md`
  - 本题 `STATUS.md` / `path-ledger.md` / `dev-log.md` / 本 EXP
- **命令**：
  ```bash
  # 语义（Python 3.12；系统 python3.10 缺 numpy）
  /usr/local/python3.12.13/bin/python3 \
    people/yuhongming-2026/MultilabelMarginLoss_2327/workdir/code/tests/local_semantic_check.py

  source /usr/local/Ascend/cann-9.1.0/set_env.sh
  cd people/yuhongming-2026/MultilabelMarginLoss_2327/workdir/code
  DEVICE_ID=0 bash build.sh -j4          # 三 key 编译
  # 首轮 -e 在 sample.py 因系统 python 缺 numpy 中断；装好后：
  DEVICE_ID=0 bash examples/run.sh
  ```
- **环境**：Ascend 910B4 · CANN 9.1.0 · `DEVICE_ID=0` · 分支 `main`

## 结果

- **功能 / 精度（样例）**：
  ```text
  Sample data generated.
  PASS: y.bin, mismatched=0/1
  PASS: is_target.bin, mismatched=0/24
  PASSED
  ```
- **性能**：未测（本闸只验证样例功能/精度；95% 门禁未开）。
- **原始日志目录**：`explorations/logs/EXP_M003_20261104_local_e/`
  - `ANALYSIS.txt`
  - `build_compile_summary.log`
  - `build_e_full.log`（run 段 PASS 摘要）

## 日志（硬性）

- [x] 终端/跑样输出归档（`build_e_full.log`）
- [x] 编译摘要归档（`build_compile_summary.log`；本 EXP 编译成功）
- [x] 样例比对结果写入 log（0 mismatch；非失败场景无需 golden dump）
- [ ] CodeArts 云端完整 `-e` tar（**下一闸**；本 EXP 为本地 910B4）

## 结论与给后人的一句话

- **采纳**：yuhongming 座位以 hongwei tip 为同源，本地样例已打通；云同步优先该座位或 hongwei `22ec6d6` tip。
- **不要再**：在多核 `rowLoss`/`y` 上用 `SetValue`/`GetValue` 写 GM（EXP_M001 已证 Cache Line 互踩）；不要给 `ScanPrefix` 加 `const`（CANN 9.1 `TBuf::Get` 非 const）。
- **重开条件**：仅当云 `-e` 或泛化 case 再 WA，且有新日志；禁止无新证据回退到标量写 GM。
