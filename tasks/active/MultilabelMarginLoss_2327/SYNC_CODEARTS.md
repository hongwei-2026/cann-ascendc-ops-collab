# CodeArts 同步与题目隔离 — MultilabelMarginLoss_2327

截图里 `/mnt/workspace` 根目录堆满 `EXP_P0xx` / `prelu2329` / tar，**会串题**。按下面收拢。

## 1. 推荐目录（云端）

```text
/mnt/workspace/
  cann-ascendc-ops-collab/          # 唯一 git clone（协作仓）
  problems/
    MultilabelMarginLoss_2327/      # 本题运行区（只放 2327）
      code/                         # ← 从座位 workdir/code 同步
      build/                        # build.sh 产物（可删）
      explorations/                 # 本题 EXP_*_logs 只进这里
    PReluGradUpdate_2329/           # 另一题，互不覆盖 code/
      code/
      explorations/
  _archive/                         # 根目录历史 EXP_*/tar 挪进来，勿再往根扔
```

## 2. 同步命令（本题）

默认座位（hongwei 初版融合）：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull
PROB=/mnt/workspace/problems/MultilabelMarginLoss_2327
mkdir -p "$PROB/code" "$PROB/explorations"
rsync -a --delete \
  people/hongwei-2026/MultilabelMarginLoss_2327/workdir/code/ \
  "$PROB/code/"
cd "$PROB/code"
bash build.sh -j4
DEVICE_ID=0 bash build.sh -j4 -e
```

队友座位改为：`people/yuhongming-2026/MultilabelMarginLoss_2327/workdir/code/`。

## 3. 硬隔离

| 做 | 不做 |
|----|------|
| 2327 的 log/tar 放 `problems/MultilabelMarginLoss_2327/explorations/` | 往 `/mnt/workspace` 根丢 `EXP_*` |
| 2329 的 log 放 `problems/PReluGradUpdate_2329/explorations/` | 用 `prelu2329/code` 覆盖 2327 |
| 协作仓台账写 `tasks/active/MultilabelMarginLoss_2327/` | 改别人的 `people/<他人>/` |
| 下载云 log 归档回本地本题 `explorations/logs/` 再 push | 只留截图 |

## 4. 根目录清理（一次性）

```bash
cd /mnt/workspace
mkdir -p _archive/prelu_exps
# 把历史 PRelu 实验挪走（按需改名）
mv EXP_P0*_logs* EXP_P100* EXP_P102* build_e*.log prelu2329 _archive/prelu_exps/ 2>/dev/null || true
```

清理后根上应只剩：`cann-ascendc-ops-collab/`、`problems/`、`_archive/`。
