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

规范见 `AGENTS.md` 探索落盘、`people/shared-lessons/general/codearts-and-logs.md`。
