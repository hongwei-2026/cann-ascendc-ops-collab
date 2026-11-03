# 云测日志归档（必须进 GitHub）

每次 CodeArts `build.sh` / `-e`（**成功或失败**）都要：

1. 本机放到本题 `explorations/logs/<EXP名>/`  
   - 至少：`build_e_full.log`（或等价全文 tee）  
   - 有则：`bins/{input,output,golden}`、`SOURCE.txt`  
2. `git add` 本题 logs → `commit` → **`push origin main`**  
3. 未 push 视为未归档。

当前已入库：

| 目录 | 结果摘要 |
|------|----------|
| `EXP20261102_161116/` | Build OK，WA 95/192 |
| `EXP_P015_.../` | WA 97/192 |
| `EXP_P016_.../` | WA 97/192 |
| `EXP_P017_.../` | SyncFunc 编译失败 |
| `EXP_P018_.../` | 仍 SyncFunc；SOURCE 写 925ea28 但编译正文是旧文件，确认同步链路失效 |
| `EXP_P019_.../` | verify PASS；bf16 Cast 炸 FATBIN（路径见本目录） |
| `EXP_P023_.../` | P023 仍 `not support bf16 type cast` → P024 关 bf16 key |
| `EXP_P024_.../` | P024 仍炸；hash=旧 fp16/bf16 → P025 仅 fp32 |
| `EXP_P025_.../` | **PASS** Build + dx/update 0/192 |
| `EXP_P026_.../` | 三 dtype 仍 bf16 Cast → P027 关 bf16 只留 fp16 |



规范见 `AGENTS.md` 探索落盘、`people/shared-lessons/general/codearts-and-logs.md`。
