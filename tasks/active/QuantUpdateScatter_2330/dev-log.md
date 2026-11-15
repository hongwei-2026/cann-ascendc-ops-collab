# 开发日志 — QuantUpdateScatter_2330

## 2026-11-15 — Q001 本地恢复开始

- 用户要求只在本地修改；本轮不云测、不 push。
- 已读仓库 AGENTS、协作规则、性能门禁、座位认领和原始代码；指定座位已认领。
- 当前协作仓快照缺少本题，不能把交接描述当成本地已有实现；只在允许的本题目录恢复。
- 原始 reference.py 注释要求 fp32，实际使用 fp64；纳入 Q001 本地正确性检查。
- 此时尚未修改内核，未运行 CANN/NPU，不写 PASS。
## 2026-11-15 — Q001 本地候选完成

- 已恢复本题源码到指定座位；未复制 build/缓存，未改原始 ZIP/解压目录或其它题。
- Q001：多核对齐拷贝、copy 核数保留、indices 缓存与回退、按 tile 复用 scales/zp；保留 RINT 转换链和 SyncAll。
- 修正 reference 的 fp32/reciprocal 语义、kernel UT 的 MODE/完整 tiling，并增加待 CANN 运行的多核双 dtype UT。
- 实测：9 项 CPU 回归、576 组调度对拍、参考探针完成；源码范围检查通过，完整日志在 explorations/logs/EXP_Q001/。
- 边界：CANN 编译、NPU 精度、TBE/profile 未运行，不记设备 PASS；用户要求仅本地，未 push。

## 2026-11-15 — 云端冒烟 PASS（Q001）

- hidevlab：`git pull` → `6a7b849`；`bash build.sh -j8 -e`
- 编译成功（ascend910b 两 key opc Done）；样例 `PASS: var_out.bin, mismatched=0/64` / `PASSED`
- `set_env.sh` 默认路径不存在，但镜像内 CANN 已可用
- 证据：`explorations/logs/EXP_Q001/C001-cloud-smoke-paste.log`；**非**性能门禁结论

## 2026-11-15 — R001（平台测点2 ACL 161002）

- 用户截图：测点1 Pass 5.89us；测点2 Runtime Error；后续 Skipped；ACL 161002（PARAM_INVALID）
- Codex 本地修复：OpDef 补齐 4 签名；kernel 按 `indicesIsInt64` 分发；tiling 拒非法 indices；示例带 `aclGetRecentErrMsg`
- 未改 Q001 计算头；平台第2点根因待重测确认；证据 `explorations/EXP_R001-runtime.md`

## 2026-11-15 — R001 云端 `-e` PASS

- `ad76638`：编译出 **4** 个 ascend910b bin；样例 `mismatched=0/64` / PASSED
- tee 目录未建导致 `/tmp/.../build_e.log` 失败；全文由粘贴归档 `logs/EXP_R001/build_e-cloud-paste.log`
- 下一闸：重新上传平台验证测点2

## 2026-11-15 — 平台 1345715 Pass + Q002 候选

- 提交 Pass（50 点）；可见点仍慢于榜最优（约 1.5–2.8×）
- Codex Q002：TQue 两槽路径 + workBuf 大块拷贝；本地 512 组类模型通过；收益未测
- Review：`TQue<...,1>` 与 InitBuffer(2) 不一致 → 主仓改为深度 **2** 后再上云
- 证据：`explorations/EXP_Q002-pipeline.md`、`decisions/DEC_Q002-pending-platform.md`

## 2026-11-15 — Q002 云端 `-e` PASS

- `796c82e`：编译 opc×4；样例 `mismatched=0/64` / PASSED
- 归档：`explorations/logs/EXP_Q002/build_e-cloud-paste.log`
- 下一闸：上传平台全 50 点，对比提交 1345715 可见时延

## 2026-11-15 — Q002 平台改善 + Q003 候选

- Q002 平台可见点 5.33/5.74/6.52/5.67/7.83us（仍慢于榜最优）
- Codex Q003：fp32 免 scalesRaw、无 zp 免 zp 缓冲；host 按实际预算抬 ubFactor；计算/队列未改
- Review：再次把 `TQue` 深度从 1 改回 **2**；证据 `explorations/EXP_Q003-ub.md`

## 2026-11-15 — Q003 云端 `-e` PASS

- `ca79bd1`：opc×4；样例 `mismatched=0/64` / PASSED
- 归档：`explorations/logs/EXP_Q003/build_e-cloud-paste.log`
- 下一闸：上传平台，对比 Q002 可见点 5.33/5.74/6.52/5.67/7.83

## 2026-11-15 — Q003 平台无收益 → 回退 Q002

- 平台 Pass；可见点 5.51/5.84/6.68/5.46/7.68（相对 Q002 噪声级涨跌）
- 归档：`logs/EXP_Q003/platform-q003-vs-q002.json`、`platform-q003-pass.png`
- DEC：回退 `tiling.cpp` + `quant_update_scatter.h` 至 `796c82e`；Q003 废弃
- 下一闸：新假设（建议 profile / 真双缓冲），禁止再抠 UB tile

## 2026-11-15 — Q005 本地采集工具准备完成

- 先登记新 DEC/道路；尊重 Q003 废弃和 Q002 回退，不改生产 host/kernel 或 TQue 深度。
- 新增独立 profile_case 与数据生成/结果校验/采集归档入口，26 组诊断覆盖同址/异址、四签名、宽窄多行、zp/reciprocal；非官方 50 点。
- 26×3 actual-runner CPU 接口模型、R001 源码契约和 shell 语法检查通过；缺失/空/歧义导出不补零、不判收益。
- 旧 Q002 CPU 回归因主仓缺少 baseline 文件未运行成功，失败日志已归档；CANN/NPU/profile/性能收益全 NOT_RUN。
- 下一闸：按本题 code/tools/README_Q005.md 采设备原始证据，再立有界候选。遵用户要求仅本地，未 commit/push。

## 2026-11-15 — Q005 Cursor review → 入库

- 确认生产 `op_host`/`op_kernel` 相对 `2d3c61c` 无改；`TQue<...,2>` + `InitBuffer(...,2)` 一致
- 工具链口径保守（COMPLETE/RAW_ONLY/INCOMPLETE），未假 PASS；inplace 多轮幂等可对齐 golden
- 台账止损改为「无 profile 不改生产」；推送 main 供云采
