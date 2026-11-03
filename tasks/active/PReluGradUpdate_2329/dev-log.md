# 开发流水 — PReluGradUpdate_2329

## 2026-03-22

- 做了：建立协作规范仓；登记本题任务卡。
- 结果：规范与分类就绪。
- 关联道路：P001

## 2026-03-22（材料 + 多路分析）

- 做了：从 `D:\华为算子比赛\` 同步并解压 ZIP；通读 Host/Kernel/reference；写入多路分析规范；完成本题 AI-1；拉起 AI-2/AI-3。
- 结果：语义金标准明确；模板 Kernel/Tiling 为空；bf16 key 有风险；update 不 reduce。
- 卡住：当时 SUMMARY 未齐。
- 下一步：当时为收齐 AI-2/3。
- 关联道路：P002、P003

## 2026-03-22（SUMMARY 定稿）

- 做了：AI-2（行片流式）、AI-3（带宽/泛化）落盘；误放根目录的 AI-3 迁入 `analysis/`；汇总裁决切分主路径与废弃清单。
- 结果：`analysis/SUMMARY.md` 已定稿；P003 采纳。
- 下一步：按 SUMMARY 写本题 `design.md`（仍不串题、不提前死磕性能）。
- 关联道路：P003、P004

## 2026-03-22（AI 工作规范）

- 做了：新增 AI 流程规范、AGENTS.md、会话卡片模板、Cursor always-apply 规则；CONTRIBUTING/README 挂接。
- 结果：后续 AI 须按阶段门禁与路径围栏工作。
- 下一步：用会话卡片指挥写 `design.md`（P004）。
- 关联道路：P005

## 2026-11-02（合并本地 Codex/Claude + 云同步）

- 做了：发现 `D:\华为算子比赛\PReluGradUpdate_problem_2329_plan`；导入 design_final；重写 SUMMARY；design.md 对齐；写 SYNC_CODEARTS；建 workdir。
- 结果：实施主依据=本地设计（Select/Inf/模式R/weights[1|C]）。
- 下一步：按 design 改 workdir Host/Kernel → push → CodeArts pull。
- 关联道路：P006、P007

## 2026-11-02（EXP001 WA → P0 实现）

- 做了：记录空 Kernel `-e` WA；实现 InferShape/Tiling/Kernel（Compare+Select，key 0/1/2）。
- 结果：已 push workdir；待云端复测。
- 下一步：pull 后 `bash build.sh -j4` + `-e`；若 WA 查 Select 极性。
- 关联道路：P008、P009

## 2026-11-02（EXP002 bf16 Cast）

- 做了：云编报 `not support bf16 type cast`；记 EXP002；bf16 改位转换，禁止 Cast。
- 结果：修复已推；待重编。
- 下一步：pull → 清 build → `bash build.sh -j4` → `-e`。
- 关联道路：P010、P011

## 2026-11-02（EXP003 Select 极性）

- 做了：`-e` 编过但 dx/update 各错 ~97/192；按半错假设交换 Select 参数。
- 下一步：pull 后重编 `-e`；若仍错再查权重广播。
- 关联道路：P012

## 2026-11-02（归档云日志 EXP20261102_161116）

- 做了：本机 `D:\华为算子比赛\EXP20261102_161116_logs.tar.gz` 解压进 `explorations/logs/EXP20261102_161116/`；更新 SYNC E2 无密码路径说明。
- 结果：`build_e_full.log` 显示 Build OK，样例仍 `Wrong Answer` dx/update 各 95/192。
- 下一步：确认云端已 `git pull` 含 f89dd20 后清 build 再 `-e`；若仍半错则查权重广播/Compare 语义。
- 关联道路：P012、P013

## 2026-11-02（EXP004 分析）

- 做了：解压 tar；用 seed=42 复算正负个数与各类错误模型错数；对照 EXP003 前后。
- 结果：95/95 = 恒 `dx=g`/`update=0`；换 Select 只是从恒负支(97)换成恒正支(95)。根因=**mask 未生效**，P012 废弃。
- 下一步：P015 修 Compare/Select mask（恢复正确 Select 序）。
- 关联道路：P014、P015

## 2026-11-02（P015 mask 修复）

- 做了：`CompareScalar` + `PipeBarrier`；Select 恢复 `g/wg`、`0/xg`；maskBuf 按 256-bit 对齐。
- 下一步：云端 pull 清 build 后 `-e`；打包日志须含 bins。
- 关联道路：P015

## 2026-11-02（P016 Codex 优化合入）

- 做了：同步 Codex `design_final`（单缓冲、coef 21/29、usable 80%、Select 极性说明）；改 workdir Kernel/Tiling；归档 EXP_P015 日志。
- 下一步：云端 `-e` 看是否仍 97；mask 根因另开路径。
- 关联道路：P016

## 2026-11-02（P017 Claude Select 别名）

- 做了：从题包模板合入 Claude 完整重写（独立 neg/fg、cmp 64 对齐、DataCopyPad）；写 EXP005。
- 下一步：云端 `-e`；若过样例再处理 bf16 Cast（EXP002）。
- 关联道路：P017

## 2026-11-02（P019 源码指纹门禁）

- 做了：Codex 加 `verify_codearts_source.sh`（比对 workdir vs prelu2329/code SHA、禁残留 SyncFunc）；修脚本里错误转义的 `${`；更新 SYNC。
- 下一步：云端必须校验 PASS 再 `-e`。
- 关联道路：P019

## 2026-11-02（P020 / EXP_P019）

- 做了：归档 EXP_P019；确认 verify PASS；bf16 去 Cast（位扩展/RNE）；写 EXP007。
- 路径：`explorations/logs/EXP_P019_20261102_175158/`
- 下一步：云端重编 `-e`。
- 关联道路：P020

## 2026-11-02（P021 试交工程包）

- 做了：从 hongwei `workdir/code`（tip `beaa00e`）打官方结构 ZIP；写 `deliverables/README.md`；gitignore 放行 `tasks/**/deliverables/*.zip`。
- 路径：`tasks/active/PReluGradUpdate_2329/deliverables/PReluGradUpdate_2329_code_best.zip`；桌面同名副本。
- 结果：云上历史最好仍 WA~95–97；本包含未复测的 bf16 位转换。
- 下一步：平台试交；同时 CodeArts 复测 P020。
- 关联道路：P021

## 2026-11-03（P023 吸 P022 / 避八参 Select）

- 做了：审 PR#1；hongwei Kernel 改 `Compare(+zero)` + **count** Select；写 EXP009；重打包试交 ZIP；PR 留言。
- 路径：`people/hongwei-2026/.../op_kernel/p_relu_grad_update.h`；`deliverables/PReluGradUpdate_2329_code_best.zip`
- 下一步：平台重交；CodeArts `-e`。
- 关联道路：P022、P023

## 2026-11-03（P024 / EXP_P023）

- 做了：归档 EXP_P023；确认仍 bf16 Cast；`DtypeBridge` 全特化；默认 `PRELU_ENABLE_BF16=0`（def/TPL/tiling/entry）。
- 路径：`explorations/logs/EXP_P023_20261103_123326/`
- 下一步：云端重编 `-e` 验样例 mask。
- 关联道路：P024

## 2026-11-03（P025 / EXP_P024）

- 做了：归档 EXP_P024；发现关 bf16 后仍炸，hash 仍是旧 fp16/bf16；改为默认 **仅 fp32**（关 FP16+BF16，头文件无活跃 Cast/bf16）。
- 路径：`explorations/logs/EXP_P024_20261103_124542/`
- 下一步：云端 `-e`；通过后再开低精度。
- 关联道路：P025

## 2026-11-03（EXP_P025 样例 PASS）

- 做了：归档 EXP_P025；确认 Build OK + dx/update 0/192；刷新试交 ZIP；写 EXP012。
- 路径：`explorations/logs/EXP_P025_20261103_125446/`；桌面 `PReluGradUpdate_2329_code_best.zip`
- 下一步：平台试交或开 fp16/bf16。
- 关联道路：P025

## 2026-11-03（合并 PR#1 yuhongming）

- 做了：Owner 合并 PR#1；保留 hongwei 台账/试交包（P025 PASS）；合入 `people/yuhongming-2026/` + EXP008 + CLAIM 已认领。
- 冲突裁决：共享 STATUS/ledger/deliverables 以 main（P025）为准。
- 下一步：队友可在自己座位继续；默认试交仍用 hongwei P025 包。
- 关联道路：P022

## 2026-11-03（P026 / 平台 507035）

- 做了：判读 ACL 507035；重开 fp16/bf16 且禁 Cast；修 weight 每 tile 重建；workspace 16MB；写 EXP013。
- 下一步：云端三 key 编过 + `-e`；再交平台。
- 关联道路：P026

## 2026-11-03（P027 / EXP_P026）

- 做了：归档 EXP_P026；确认仍 bf16 Cast；改为只开 fp32+fp16、关 bf16；写 EXP014。
- 路径：`explorations/logs/EXP_P026_20261103_130902/`
- 下一步：云端两 key `-e`。
- 关联道路：P027

## 2026-11-03（P028 / EXP_P027）

- 做了：归档 EXP_P027；确认 half 同 TU 导致 fp32/fp16 都 Cast 炸；回到仅 fp32；修 verify 误报；写 EXP015。
- 路径：`explorations/logs/EXP_P027_20261103_131350/`
- 下一步：平台继续交 fp32 包；低精度另开独立实现。
- 关联道路：P028

## 2026-11-03（P029）

- 做了：根因定为 DtypeBridge/ToFloat 显式特化污染各 key；改原生 float/half + bf16 仅 if constexpr 位转换；开三 dtype；写 EXP016。
- 下一步：云端 verify + `build.sh -j4 -e`，下载日志归档。
- 关联道路：P029

## 2026-11-03（P029 云测 PASS）

- 做了：归档 EXP_P029；三 key Done + 样例 0/192 PASS；刷 `code_best.zip`。
- 路径：`explorations/logs/EXP_P029_20261103_132332/`
- 下一步：平台重交。
- 关联道路：P029

## 2026-11-03（P030 / 平台 case4 仍 507035）

- 做了：判读非「缺注册」；half 改 Cast↔fp32；bf16 补 S_V/S_MTE3；写 EXP017。
- 下一步：云端 `-e` 后平台重交。
- 关联道路：P030

## 2026-11-03（队友同步）

- 做了：写 `TEAM_NOTICE.md`；STATUS/README 置顶；shared-lessons L08。
- 下一步：队友 pull main 读 TEAM_NOTICE；hongwei 继续 P030 云测。
- 关联道路：P030

## 2026-11-03（P030 日志 + P031 Codex）

- 做了：归档 EXP_P030（云测 PASS、平台仍 507035）；采纳 Codex：weight Duplicate 对齐（EXP018/P031）；本地语义 PASS。
- 下一步：云端 `-e` P031 → 平台重交。
- 关联道路：P031

## 2026-11-03（平台 1–5 Pass）

- 做了：确认 P031 后 case4/5 过；1–5 全 Pass；6 隐藏；用时 ms 级 vs 最优 μs。
- 下一步：开性能道路（有界）；功能对齐逻辑冻结。
- 关联道路：P031 采纳

## 2026-11-03（P100 review）

- 做了：审 Codex 性能改；修 Process 跳步 WA；host tile 限 inner；更新 EXP019/语义检查。
- 下一步：CodeArts `-e` → 平台采时。
- 关联道路：P100

## 2026-11-03（EXP_P100 云测 PASS）

- 做了：归档 EXP_P100；三 key Done；样例 0/192 PASS。
- 路径：`explorations/logs/EXP_P100_20261103_144306/`
- 下一步：平台交 P100 采时。
- 关联道路：P100

## 2026-11-03（审 P032 + 性能阶段通告）

- 做了：对照 yuhongming P032 vs P100；更新 TEAM_NOTICE/STATUS（宣布性能阶段）；写 EXP020。
- 下一步：平台采时；队友勿交八参 Select 包。
- 关联道路：P032 部分采纳 · P100

## 2026-11-03（EXP021 拒收外部模板回归）

- 做了：审 `D:\华为算子比赛\...\template\code` 15:03 kernel；相对 P100 为回归；记 P101 废弃。
- 下一步：仍交平台采 P100；Codex 新产物须落 `people/hongwei-2026/` 再审。
- 关联道路：P101 废弃 · P100

## 2026-11-03（复审 P102）

- 做了：确认对齐 `DataCopy` 候选可试；补 EXP022 复审与语义指纹；提醒 B 类 ≤2h 止损。
- 下一步：CodeArts `-e` 归档后再平台同机复测。
- 关联道路：P102

## 2026-11-03（EXP_P102 云测 PASS）

- 做了：归档 `EXP_P102_20261103_154833`；三 key Done；样例 0/192 PASS；说明 CRLF/LF hash 差。
- 下一步：平台交 P102 采时。
- 关联道路：P102




## 2026-11-03（P031 / case4 UB 地址对齐）

- 做了：审查平台 507035 的全部向量入口；发现 per-channel 权重展开使用 `Duplicate(weightLocal[localOffset], ...)`，当 `innerSize` 不是 8 个 float 的倍数时会从非 32B 对齐地址启动向量指令。
- 修复：每个 channel segment 的未对齐前缀改用 `SetValue`，只在 `fillOffset % 8 == 0` 后调用 `Duplicate`，进入 `Mul` 前补 S_V 同步。
- 验证：`python tests/local_semantic_check.py` PASS；新增 innerSize=1/3/5/7/35/8255 和非零 global start 对齐回归；`git diff --check` PASS。
- 下一步：CodeArts 三 key 编译与 `-e`，归档日志后平台重交。
- 关联道路：P031、EXP018

## 2026-11-03（P032 / yuhongming 座位吸 P031）

- 做了：按 EXP018 与报错台账，把对齐修复合入 `people/yuhongming-2026/.../BuildWeightVector`；补 `check_weight_fill_alignment`；本地语义 PASS。
- 下一步：CodeArts 同步自己座位 `workdir/code` 后 `-e`。
- 关联道路：P032、EXP018

## 2026-11-03（P100 / 首轮性能优化）

- 平台基线：1–5 全 Pass；本实现 1.30/1.32/2.00/1.79/1.79ms，参考约 9.82–13.76μs。
- 定位一：P026 遗留 16MB workspace，但 Kernel `(void)workspace`，形成与 tensor 大小无关的固定开销；恢复为 0。
- 定位二：公开 fp32 `[4,6,8]` 被 tiling 成 1 核，并串行处理 24 个 channel segment；改为 32B block 全核切分（24 核）。
- Kernel：每 tile 限制在单个 channel 内，直接加载一个权重并 `Muls`；删除 weightFloatBuf、Duplicate 展开和逐段对齐分支。
- 验证：本地数值、channel tile 边界、fp32 24 核/fp16 12 核 tiling 回归 PASS；`git diff --check` PASS。
- 下一步：CodeArts 三 key 编译 + `-e`；平台采时后再决定是否开 P101（双缓冲/半精度路径）。
- 关联道路：P100、EXP019

## 2026-11-03（P102 / 对齐 tile 连续搬运候选）

- 14:46:56 平台截图：1–5 Pass，42.41 分；本实现 7.26/7.41/7.33/5.72/5.82 μs，参考 4.13/4.16/4.23/3.99/4.01 μs。截图不含源码 hash，仅作为 P100 时序对应的性能基线。
- 开 P102：GM 起点和长度均 32B 对齐时，四路搬运用连续 `DataCopy`；非对齐 tile 保留 `DataCopyPad`。本地语义检查与 `git diff --check` PASS；云端收益见 EXP022。
- 下一步：CodeArts 三 key + `build.sh -j4 -e`，归档完整日志；随后平台同机复测。不得把本地语义检查写成真机 PASS。
- 关联道路：P100、P102、EXP022

## 2026-11-03（P102 平台未见收益，恢复 P100）

- 最新 15:48:12 截图：1–5 Pass，排名 95、42.06 分，8.89/8.75/8.99/6.11/6.01 μs；上一图排名 87、42.41 分，前三点慢 18%–23%，后两点慢 6.82%/3.26%。截图已本地归档。
- 结论：P102 无正向收益证据，本轮不采纳。仅撤回 Kernel 对齐 DataCopy 分支及其 3 条专用源码断言和注释；恢复 `3bd65fb` 的 P100，保留所有历史云测与协作改动。
- 证据边界：平台时间 15:48:12 早于完整云包 SOURCE 时间 15:48:33；此前 15:44:12 包指纹是 P102 但日志不完整。截图本身无源码 hash，不能单凭图认定具体退化原因或承诺回退后的耗时。
- 下一步：同环境、同输入、同产物指纹做 P100/P102 重复 A/B 与 profile；见 DEC001。按用户要求只在本地修改，不推送。
- 本地复核：P102 原始包与完整日志确认 source/build 指纹一致、三 key Done、样例 PASS；`op_kernel_logs/` 为空，无性能统计。恢复副本语义检查 PASS，源码与检查脚本精确匹配 P100。
- 关联道路：P102、EXP022、DEC001








