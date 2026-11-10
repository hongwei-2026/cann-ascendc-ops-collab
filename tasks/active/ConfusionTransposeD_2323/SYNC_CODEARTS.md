# CodeArts 同步 — ConfusionTransposeD_2323

仓库：https://github.com/hongwei-2026/cann-ascendc-ops-collab  

本闸座位：`people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/`（P002 泛化/云测）  
勿串：`PReluGradUpdate_2329` / `MultilabelMarginLoss_2327`。

推荐云端布局（与 2327 一致）：

```text
/mnt/workspace/
  cann-ascendc-ops-collab/
  problems/
    ConfusionTransposeD_2323/
      code/            # 运行区
      explorations/    # 本题 tar / log
  exp_logs/            # 可选：再拷一份便于 IDE 下载
    ctd2323/
```

---

## A. 本机先 push（否则云端 pull 不到 P002）

PowerShell，协作仓根：

```powershell
cd "D:\sageAttention2 算子开发任务书"
git -c http.sslVerify=false pull origin main
git add "people/hongwei-2026/ConfusionTransposeD_2323" "tasks/active/ConfusionTransposeD_2323"
git status
git commit -m "ConfusionTransposeD: hongwei seat + P002 generalize/cloud harness"
git -c http.sslVerify=false push origin main
```

---

## B. CodeArts 终端 — 第一次

```bash
cd /mnt/workspace

# 1) 协作仓（已有则跳过 clone）
if [ ! -d cann-ascendc-ops-collab/.git ]; then
  git clone https://github.com/hongwei-2026/cann-ascendc-ops-collab.git
fi
cd cann-ascendc-ops-collab
GIT_EDITOR=true git pull origin main

# 2) 本题隔离运行区
ME=hongwei-2026
PROB=/mnt/workspace/problems/ConfusionTransposeD_2323
mkdir -p "$PROB/code" "$PROB/explorations" /mnt/workspace/exp_logs/ctd2323

rsync -a --delete \
  /mnt/workspace/cann-ascendc-ops-collab/people/${ME}/ConfusionTransposeD_2323/workdir/code/ \
  "$PROB/code/"

# 无 rsync 时：
# rm -rf "$PROB/code" && cp -a .../workdir/code "$PROB/code"

# 3) 环境 + 跑（路径以云环境为准）
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$PROB/code"
DEVICE_ID=0 bash tests/run_c002_cloud.sh

# 4) 日志挪到可下载位置
cp -a logs/EXP_C002_*.tar.gz "$PROB/explorations/" 2>/dev/null || true
cp -a logs/EXP_C002_*.tar.gz /mnt/workspace/exp_logs/ctd2323/ 2>/dev/null || true
ls -lt logs/EXP_C002_*.tar.gz "$PROB/explorations" /mnt/workspace/exp_logs/ctd2323 | head
```

只跑样例（不做泛化）：

```bash
cd /mnt/workspace/problems/ConfusionTransposeD_2323/code
DEVICE_ID=0 bash build.sh -j4 -e 2>&1 | tee /mnt/workspace/exp_logs/ctd2323/build_e_$(date +%Y%m%d_%H%M%S).log
```

---

## C. 以后每次（本机已 push）

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
GIT_EDITOR=true git pull origin main
ME=hongwei-2026
PROB=/mnt/workspace/problems/ConfusionTransposeD_2323
rsync -a --delete \
  people/${ME}/ConfusionTransposeD_2323/workdir/code/ \
  "$PROB/code/"
cd "$PROB/code"
rm -rf build build_out examples/build samples/build logs
DEVICE_ID=0 bash tests/run_c002_cloud.sh
cp -a logs/EXP_C002_*.tar.gz "$PROB/explorations/" /mnt/workspace/exp_logs/ctd2323/
```

---

## D. 下载日志 → 本机归档 → push

1. CodeArts 资源管理器打开  
   `/mnt/workspace/exp_logs/ctd2323/` 或 `problems/ConfusionTransposeD_2323/explorations/`  
2. 右键 `EXP_C002_*.tar.gz` → **Download**  
3. 本机解压/拷入：

```powershell
cd "D:\sageAttention2 算子开发任务书"
$dest = "tasks\active\ConfusionTransposeD_2323\explorations\logs\EXP_C002"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
# 把下载的 tar.gz 拷进 $dest，再按需 tar 解开
git add "tasks/active/ConfusionTransposeD_2323/explorations/logs"
git commit -m "ConfusionTransposeD: archive EXP_C002 cloud logs"
git -c http.sslVerify=false push origin main
```

规范：成功/失败全文 log 都要归档 push；禁止只留截图。

---

## E. 注意

- 截图环境是 **NPU:A2**；任务书写 910B。功能可先 A2 打通；合入/验收以 910B 为准。  
- 勿用 `prelu2329/code` 或 `MultilabelMarginLoss_2327` 目录跑本题。  
- 云端未 `git pull` 到含 `hongwei-2026/ConfusionTransposeD_2323` 的 commit 前，不要 build。
