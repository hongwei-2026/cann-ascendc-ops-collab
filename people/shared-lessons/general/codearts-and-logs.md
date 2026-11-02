# CodeArts 同步与日志

- 本机改完 → `git push` 公开仓 → 云端 `git pull` → **再** `cp/rsync` 覆盖题包 `code/`，禁止先 build 后 pull。  
- 合并若弹出 vim：`Esc` → `:wq`；或 `GIT_EDITOR=true git pull`。  
- 每次 `-e`：`tee` 全文 log，并打包 `output/golden/input` 进 tar；只截图 = 未归档。  
- 无密码路径：云端打 tar → IDE 右键 Download → 本机归档进 `tasks/.../<题>/explorations/logs/` → **立刻 commit + push**。  
- **编译错误、Wrong Answer、超时日志一律推 GitHub**；三人分座也要能看见彼此的失败原文，不要只放本机 Downloads。
