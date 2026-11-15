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
