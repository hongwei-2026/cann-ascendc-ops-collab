# ???? ? MultilabelMarginLoss_2327

## 2026-11-03

- **??**??? design?hongwei ?? push?CodeArts EXP_M001 ?????
- **??**??? **build OK**?`is_target` PASS?**`y` WA ? FAILED**?? `explorations/logs/EXP_M001_*`??
- **???**????? / mean Finalize?? `-e`?

## 2026-11-03 17:21

- hongwei: Claude/Codex ? EXP_M001 y WA?DMA ? + SyncAll + usedCores??commit `3ebda45` ?????????? pull/rsync/`-e`?

## 2026-11-03 17:25

- EXP_M002 ?????`ScanPrefix` const???? logs?? const ?????

## 2026-11-04

- **??**?yuhongming ???? hongwei tip?DMA + ? ScanPrefix const + classTile???? semantic PASS?910B4 `build.sh` ? key OK?`run.sh` **PASSED**??? EXP_M003 ?????? `explorations/EXP_M003-local-910b4-pass.md` ??????
- **??**??? `y`/`is_target` ? 0 mismatch?
- **??**??? `git push origin main` ? GitHub 443 ?????????
- **???**?CodeArts ? SYNC ?????? `-e`???? case?

## 2026-11-04 19:34
- 拉入 yuhongming EXP_M003（本地 -e PASS，同源 tip）。
- hongwei M006：ProcessOneRow 早退；local_semantic_check 扩 K≠C/空正类/classTile mirror；semantic PASS。
- 下一闸：云 rsync hongwei code → -e。

## 2026-11-04 19:39
- EXP_M004 CodeArts -e **PASSED**（y 0/1 · is_target 0/24）。日志已归档。

## 2026-11-04 19:51
- 平台用例 1–5 **Pass**（错误 0%）；用时约 713μs～2.1ms，最优约 4.7～5.2μs（~150×）。
- STATUS → 性能中入口；已 push 台账。


## 2026-11-04 M009 candidate
- Based on hongwei M006 / b61feb7; ZIP and source fingerprints matched.
- Removed fixed 16MB workspace; none/single-core use 0, multi-core mean/sum use aligned FP32 losses[N]. Tiny reductions finalize on one core.
- CPU semantic PASS; actual Process flow 36 cases and workspace 42 cases PASS. Generated 72 independent reference cases; NPU NOT RUN.
- Added source manifest and isolated cloud check; preserved M006 as validated baseline. CANN build, NPU precision, and same-platform performance remain pending. No commit/push.

## 2026-11-04 20:35
- 归档 EXP_M009：多核 mean/sum ACL 507035。
- M010：多核恢复 16MB 系统区 + GetUserWorkspace；none/tiny 仍 0。

## 2026-11-04 21:08
- 新日志 EXP_M009_204303：指纹仍是旧 M009，非 M010；多核依旧 507035。
- 下一闸：云端必须 git pull 到 e1b2de2+ 再 rsync。
