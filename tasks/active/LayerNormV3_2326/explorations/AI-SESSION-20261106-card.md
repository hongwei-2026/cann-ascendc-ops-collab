# AI 会话卡片 — LayerNormV3_2326（2026-11-06）

| 字段 | 内容 |
|------|------|
| 日期 | 2026-11-06 |
| 题名 | LayerNormV3_2326 |
| 当前 STATUS | 实现中（本会话新建） |
| 本会话唯一目标 | 按模板 zip 和本仓同类题目，在 yuhongming 座位实现 LayerNormV3，写下对话并推到 GitHub |
| AI 角色 | Implementer |
| 允许改动的路径 | `people/yuhongming-2026/LayerNormV3_2326/**`，`tasks/active/LayerNormV3_2326/**`，题目索引（`tasks/README.md`、根 `README.md`、`people/yuhongming-2026/README.md`） |
| 明确禁止 | 其他题目的实现、他人 `people/`、把没跑的 NPU 结果写成 PASS |
| 关联台账 ID | P001 |
| 合入仓路径 | 本协作仓 `main`；CodeArts 只同步本座位 `workdir/code/` |

## 完成定义

- [x] 模板 TODO（kernel / tiling / infershape）已填上，并与 reference 的尾维广播对齐
- [x] `local_semantic_check.py` 通过
- [ ] 910B `bash build.sh -e`（本机没有设备）

## 结束后勾选

- [x] `path-ledger` 已更新
- [x] `dev-log` 已追加
- [x] 未改其他题目的代码
- [x] 下一闸：910B 上跑 `-e`
