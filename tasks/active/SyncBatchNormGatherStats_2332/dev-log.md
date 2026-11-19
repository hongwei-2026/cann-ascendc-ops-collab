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

## 2026-11-17（EXP_S003 云 `-e`）

- 做了：云 pull→`74674f2`；`build.sh --make_clean` + `-j8 -e`；归档 EXP_S003。
- 结果：**PASS**（四输出 0/6）。
- 卡住：无。
- 下一步：按 SUBMIT_LINKS_S003 重交平台；结果回写 S003 结案或开 S004。
- 关联道路：S003
## 2026-11-17 S004

- **事实**：平台反馈仍为 Wrong Answer；截图显示共 50 点，1–5 公开 Pass，6–50 隐藏。
- **假设**：`QuantFp16Vec` 的 fp16 暂存往返可能吞掉 `count==1` 产生的 `new_variance` NaN/±Inf；S003 只覆盖最终 half 写回。
- **改动**：`op_kernel/sync_batch_norm_gather_stats.h` 在通用 `QuantFp16Vec` 的第二次 Cast 前，用 `V_S` 同步并恢复原始非有限值；有限值路径不变。
- **验证**：`git diff --check` 通过；Python reference 边界探针确认 NaN/±Inf 语义。Windows 本机无 WSL/CANN，`bash build.sh -j8 -e` 启动失败，全文记录于 `explorations/logs/EXP_S004/build_e.log`。
- **下一步**：在 910B 环境构建、归档完整云日志，更新 S004 提交清单后重交平台；不做性能优化。

## 2026-11-17（S004 代码审阅）

- 做了：认可假说；`QuantFp16Vec` 标量回写后补 `S_V`，`StoreAsInputDtype` 标量回写后补 `S_MTE3`。
- 结果：未云测；不宣称 PASS。
- 下一步：云 `-e`（EXP_S004）→ SUBMIT_LINKS_S004 重交平台。
- 关联道路：S004

## 2026-11-17（EXP_S004 云 `-e`）

- 做了：云 pull→`fa21ed5`；`build.sh -j8 -e`；归档 EXP_S004。
- 结果：**PASS**（四输出 0/6）。
- 下一步：SUBMIT_LINKS_S004 重交平台。
- 关联道路：S004

## 2026-11-19（S004 平台失败 → 回退 → S005）

- 做了：用户反馈平台仍 WA（错误同前），可见点用时更长；剥掉 S003/S004 标量保 NaN 环，Quant/Store 回 S002；写 `HANDOFF-S005-PLATFORM-WA-PROMPT.md`；台账 S004=废弃、S005=进行中。
- 结果：代码无 Cast 后全 tile GetValue/SetValue；ComputeTile 内 invstd 量化后补丁保留（S002 起已有）。
- 卡住：隐藏点根因仍未知（优先 count 语义 / W>1 / dtype / 下载工程）。
- 下一步：按 S005 HANDOFF 取证；勿重交 S004 包；回退后云 `-e` 再交平台可选。
- 关联道路：S004（废弃）· S005

## 2026-11-19（S006 直接除法边界）

- 假设：乘预先舍入的 fp32 倒数与 reference 的 fp32 除法不同，可能使结果越过 fp16 舍入边界。
- 改动：Kernel 的 `accSum` / `accSq` 两处归一化改用向量 `Div`；不改 count 语义、特殊值处理或切分。
- 验证：CPU 数值模型的固定 6 通道例子中，旧路径 `new_variance` 有 5 个值不同于 reference，直接除法模型均相同；模型并非设备测试。Python 测试脚本语法和 `git diff --check` 通过。
- 审阅（本会话）：同意假说；`load` 复用安全；reference 确为 `float32/float32` 除法；复现 `new_v` fp16 差约 0.004（> 样例 atol 1e-4）。Div 略重于 Muls，但本轮先修 WA。
- 云结果：`build.sh -j8 -e` + `s006_rounding.py` 均 **PASS** 0/6；归档 EXP_S006。
- 下一步：按 SUBMIT_LINKS_S006 重交平台；结果回写 S006 结案或开 S007。

## S007 多维精度测试

- 原 ZIP 仅发现一份首点 case.json；已有样例及 UT 固定 C=6、world=1、fp32/int32。
- 新增 157 个可复现用例，覆盖四组 dtype、通道边界、rank、count、属性及特殊值；count=0 合法性待确认。
- world>1 和多 count 为语义探针，分别列出候选公式，不冒充权威真值。
- 本机只跑测试生成器自检；910B 构建、矩阵和平台结果均待执行。
- 审阅（本会话）：**采纳为排障工具**；未改 Kernel；本机 `--self-test` PASS（157）；CMake 挂 `s007_runner`+`sample_support.h` 合理。注意：`c524289` 可能慢/吃内存；PROBE≠真值；矩阵全绿≠平台过。
- 云结果：PASS=144 / FAIL=1 / ERROR=0 / PROBE=12。PROBE 证实 Kernel=world/count 求和；唯一 FAIL=`large_f32_i32_c524289`（batch_mean 大片 0，new_mean 全对）。
- 下一步：云归档 matrix.log+results.json 并 push；开 S008 修大 C（优先 `SetGlobalBuffer` 带 numel）。

## S008 多 tile 同步排查

- 静态检查：tile 尾部已有 MTE3_MTE2，但后续 ComputeTile 先复用 VECCALC 缓存；
  单独等待 MTE2 不保证 MTE3 完成读取后才启动下一次向量操作。
- 最小改动：保留原 MTE3_MTE2，新增 MTE3_V 后再复用缓存；未改 count、world 或 GM 长度。
- 检测脚本新增按名称筛选，保留原矩阵索引和随机种子；先重复失败大 C 再全矩阵。
- 用户提供的 S007 终端粘贴记录已原样归档；独立日志文件及二进制数据尚未同步。
  未在 910B 验证 S008；SetGlobalBuffer 长度假说仍未证实。
- PROBE 对照的是自定义核与候选模型，不等于独立内置算子或平台裁决。
- 审阅（本会话）：**采纳待云验**。下一 tile `Duplicate(accSum)` 可在 `batch_mean` DMA 未完成时改写 UB，能解释「batch_mean 大片 0 / new_mean 全对」。`--case` 单复现合理。
- 云结果（d75aa78）：构建与样例 PASS；大 C 连续三次 batch_mean 0 错，invstd 7 错，new_mean 0 错，new_variance 13 错；完整矩阵严格结果 144 PASS / 1 FAIL / 12 PROBE。
- 用户提供的 S008 终端记录已归档；独立云日志文件仍应保留。

## S009 舍入与判定规则分析

- S007 脚本要求零超差；`samples/sample.py` 另允许每个输出不超过 0.0001 的超差比例。
- 大 C 的 7 / 524289、13 / 524289 均低于按样例数值外推的阈值 52.4289；平台规则未核实，不称平台 PASS。
- CPU 原种子 fp32 模型与 reference 的 new_variance 有 12 个超差；日志中的前 8 个设备例值均与模型相同。7 个 invstd 差异未由 CPU 模型复现。
- 矩阵严格状态保留，附加显示样例比例结果；内核数值语义未改。
