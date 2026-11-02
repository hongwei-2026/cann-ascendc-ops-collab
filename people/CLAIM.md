# 座位认领

| 座位 | 目录 | GitHub | 状态 | 认领日期 | 备注 |
|------|------|--------|------|----------|------|
| 1 | `hongwei-2026/` | [hongwei-2026](https://github.com/hongwei-2026) | **已认领** | 2026-11-02 | Owner / 本题先发 |
| 2 | `yuhongming-2026/` | [yuhongming-2026](https://github.com/yuhongming-2026) | **已预留** | 2026-11-02 | 已邀请 push；登录后把状态改为「已认领」并写日期 |
| 3 | `seat-3/` | （待填） | **空缺** | — | 第三人：把目录改名为自己的 GitHub 登录名，并改本表 |

## 认领操作（第三人）

```bash
# 本地
git pull
mv people/seat-3 people/<你的GitHub登录名>
# 编辑 people/CLAIM.md：座位3 填登录名、状态=已认领、日期
# 编辑 people/<你>/README.md 署名
git add people
git commit -m "people: claim seat-3 as <登录名>"
git push
```

冲突时以 **先 push 到 main 的 CLAIM 行为准**；被占则改认另一空位或协调 Owner。
