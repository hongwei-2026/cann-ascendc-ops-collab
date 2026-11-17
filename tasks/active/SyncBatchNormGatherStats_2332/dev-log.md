# 开发流水 — SyncBatchNormGatherStats_2332

## 2026-11-17

- 做了：读 AGENTS / 05-ai-work-protocol / CLAIM；启用 cann-spmv-ascendc；从 `D:\华为算子比赛\SyncBatchNormGatherStats_2332.zip` 导入代码到 `people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code/`（剔 `build/`、`__pycache__`、stub_reg.log）；建 STATUS / path-ledger / CLOUD-RUN / AI-HANDOFF / TASK / WORKDIR；三路独立分析 AI-1/2/3 + `analysis/SUMMARY.md` 定稿。
- 结果：座位 yuhongming-2026；基线完整；SUMMARY 共识=验证优先勿重写；废弃清单 D-01～D-10；STATUS=`分析定稿`。
- 卡住：本机 `git pull` SSL 失败（未阻塞）；云测未跑；未 commit（等用户下令）。
- 下一步：用户确认「可以编码/云测」→ P0 `build.sh -e` + 边界对拍；公开时 commit + push main。
- 关联道路：S001（结案）· S002（规划中）

## 2026-11-17（S002 开工）

- 做了：STATUS→实现中（P0 云冒烟）；S002 标进行中；CLOUD-RUN 改为 EXP_S002；准备 commit/push 供云端 pull。
- 结果：本机不改 Kernel；等云端 `-e` 全文 log。
- 卡住：无（等云结果）。
- 下一步：云端跑完后把 `/tmp/sbngs2332_s002/build_e.log` 归档到 `explorations/logs/EXP_S002/` 并 push。
- 关联道路：S002

## 2026-11-17（S002 结案）

- 做了：云机 `reset --hard origin/main`→`219e221`；`build.sh -j8 -e`；归档 EXP_S002。
- 结果：**PASS**（batch_mean/invstd/new_mean/new_variance 均为 mismatched=0/6；末行 PASSED）。`ascend-toolkit/set_env.sh` 不存在，构建仍过（镜像已有 CANN）。
- 卡住：无。
- 下一步：全量/平台精度或边界补测；不开性能；勿把样例当 50 点。
- 关联道路：S002

## 2026-11-17（平台 WA → S003）

- 做了：平台提交 S002 七文件；结果 Wrong Answer；写 `HANDOFF-PLATFORM-WA-PROMPT.md` 交其他 AI。
- 结果：测点 1–5 Pass/0%；6 起 `*`；用时 ~5.9–6.2μs（最优 ~3.7，本轮不追）。
- 卡住：隐藏点无明细；高危假说见 SUMMARY（count 求和、W>1、fp16 二次 Cast 吞 NaN 等）。
- 下一步：新对话按 HANDOFF 排障；最小修后重交。
- 关联道路：S003

## 2026-11-17（S003 fp16 特殊值探针）

- 官方任务书将 `sampleCount` 描述为各设备计数；保留现有 count 求和与
  `worldSize` 归约语义。
- 仅修改 `op_kernel/sync_batch_norm_gather_stats.h`：最终 fp16 向量 Cast 后，
  在 DMA 前重新写回源 NaN 与 +/-inf。
- `git diff --check` 通过；reference 负 `var+eps` 探针确认
  `batch_invstd=NaN`。
- 云端构建与平台重交尚未执行，不宣称 PASS。
