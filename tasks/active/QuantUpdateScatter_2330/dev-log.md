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
