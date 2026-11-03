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




