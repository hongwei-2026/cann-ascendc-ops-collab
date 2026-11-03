# CodeArts 同步（无 IDE 远程时用 git，勿网页狂传）

仓库：https://github.com/hongwei-2026/cann-ascendc-ops-collab  

**个人实现目录**（三人隔离，见 `people/CLAIM.md`）：

```text
people/<你的GitHub登录名>/PReluGradUpdate_2329/workdir/code/
```

hongwei 当前路径：`people/hongwei-2026/PReluGradUpdate_2329/workdir/code/`  
共享台账仍在：`tasks/active/PReluGradUpdate_2329/`（无 workdir，见 `WORKDIR.md`）

---

## A. 本机（Windows PowerShell）— 改完就推

在协作仓根目录：

```powershell
cd "D:\sageAttention2 算子开发任务书"
git add -A
git status
git commit -m "PReluGradUpdate: update workdir / docs"
git -c http.sslVerify=false push origin main
```

若只推本题：

```powershell
git add "tasks/active/PReluGradUpdate_2329"
git commit -m "PReluGradUpdate: sync task materials"
git -c http.sslVerify=false push origin main
```

---

## B. CodeArts 云端终端（`/mnt/workspace`）— 第一次

```bash
cd /mnt/workspace

# 1) 拉协作仓（含规范 + workdir 代码）
git clone https://github.com/hongwei-2026/cann-ascendc-ops-collab.git

# 2) 解压官方题包（你 ls 里已有 zip）
mkdir -p prelu2329 && unzip -o PReluGradUpdate_problem_2329_template.zip -d prelu2329

# 3) 用协作仓里的实现覆盖题包 code（只覆盖工程，不碰无关目录）
# 把 hongwei-2026 换成你自己的 people 目录名
ME=hongwei-2026
rsync -a --delete \
  cann-ascendc-ops-collab/people/${ME}/PReluGradUpdate_2329/workdir/code/ \
  prelu2329/code/

# 4) 强制校验实际构建树，防止“提交已更新、prelu2329/code 仍是旧副本”
bash cann-ascendc-ops-collab/people/${ME}/PReluGradUpdate_2329/verify_codearts_source.sh \
  /mnt/workspace/cann-ascendc-ops-collab /mnt/workspace/prelu2329/code

# 5) 只有校验 PASS 后才构建（按题包 README；环境先 source cann）
cd prelu2329/code
# source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 路径以云环境为准
bash build.sh -j4
# DEVICE_ID=0 bash build.sh -j4 -e
```

若无 `rsync`：

```bash
ME=hongwei-2026
rm -rf /mnt/workspace/prelu2329/code
cp -a /mnt/workspace/cann-ascendc-ops-collab/people/${ME}/PReluGradUpdate_2329/workdir/code \
      /mnt/workspace/prelu2329/code

bash /mnt/workspace/cann-ascendc-ops-collab/people/${ME}/PReluGradUpdate_2329/verify_codearts_source.sh \
  /mnt/workspace/cann-ascendc-ops-collab /mnt/workspace/prelu2329/code
```

---

## C. CodeArts — 以后每次同步（本机已 push）

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
GIT_EDITOR=true git pull origin main

ME=hongwei-2026   # ← 改成你的登录名
rm -rf /mnt/workspace/prelu2329/code
cp -a /mnt/workspace/cann-ascendc-ops-collab/people/${ME}/PReluGradUpdate_2329/workdir/code \
      /mnt/workspace/prelu2329/code

bash /mnt/workspace/cann-ascendc-ops-collab/people/${ME}/PReluGradUpdate_2329/verify_codearts_source.sh \
  /mnt/workspace/cann-ascendc-ops-collab /mnt/workspace/prelu2329/code

cd /mnt/workspace/prelu2329/code
bash build.sh -j4
```

---

## D. 若 GitHub 私有仓 clone 要登录

CodeArts 上：

```bash
# 用 HTTPS + Token（GitHub Settings → Developer settings → PAT，勾 repo）
git clone https://<你的用户名>:<PAT>@github.com/hongwei-2026/cann-ascendc-ops-collab.git
```

或先：

```bash
git config --global credential.helper store
git clone https://github.com/hongwei-2026/cann-ascendc-ops-collab.git
# 按提示输入用户名和 PAT
```

---

## E. 日志下载 / 归档（纠错必做）

规范：每次探索除截图外，**必须留下原始 log 文件**。  
**硬规则（2026-11-03 起）**：云端 `build.sh` / `-e` **无论成功失败**，都必须打 `EXP_*_logs.tar.gz` → IDE 右键 Download → 本机归档进 `explorations/logs/` 再 push。禁止只说「又报错了」不给 log。

**目录约定（2026-11-03 起）**：云端日志**禁止**再堆在 `/mnt/workspace/` 根下。统一进：

```text
/mnt/workspace/exp_logs/prelu2329/
  EXP_P103_YYYYMMDD_HHMMSS_logs/
  EXP_P103_YYYYMMDD_HHMMSS_logs.tar.gz
```

### E0. 一次性清理（根目录已堆满 EXP_* 时）

在 CodeArts 终端执行。只动 `EXP_*` 垃圾，**不动** `cann-ascendc-ops-collab` / `prelu2329` / `build` / `data`。

```bash
cd /mnt/workspace
mkdir -p exp_logs/prelu2329/_old_root_dump_$(date +%Y%m%d)

# 先挪走（可回滚）；确认本机已下载过的再删
shopt -s nullglob
for x in EXP_*_logs EXP_*_logs.tar.gz EXP_*_build_e.log; do
  [ -e "$x" ] || continue
  mv "$x" exp_logs/prelu2329/_old_root_dump_$(date +%Y%m%d)/
done
shopt -u nullglob
ls -la
ls exp_logs/prelu2329/_old_root_dump_$(date +%Y%m%d) | head

# 本机已全部归档后，可彻底删旧堆（慎用）：
# rm -rf exp_logs/prelu2329/_old_root_dump_*
```

### E1. CodeArts：打包到 `exp_logs/prelu2329/`（IDE 右键 Download）

```bash
LOG_ROOT=/mnt/workspace/exp_logs/prelu2329
EXPNAME=EXP_P103_$(date +%Y%m%d_%H%M%S)   # 按道路改名
OUT=${LOG_ROOT}/${EXPNAME}_logs
mkdir -p "$OUT"

REPO=/mnt/workspace/cann-ascendc-ops-collab
ME=hongwei-2026
{
  echo "date=$(date -Iseconds)"
  echo "repo_commit=$(git -C "$REPO" rev-parse HEAD)"
  echo "source_kernel=$(sha256sum "$REPO/people/$ME/PReluGradUpdate_2329/workdir/code/op_kernel/p_relu_grad_update.h" | awk '{print $1}')"
  echo "build_kernel=$(sha256sum /mnt/workspace/prelu2329/code/op_kernel/p_relu_grad_update.h | awk '{print $1}')"
} | tee "$OUT/SOURCE.txt"

bash "$REPO/people/$ME/PReluGradUpdate_2329/verify_codearts_source.sh" \
  "$REPO" /mnt/workspace/prelu2329/code | tee -a "$OUT/SOURCE.txt"

cd /mnt/workspace/prelu2329/code
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
DEVICE_ID=0 bash build.sh -j4 -e 2>&1 | tee "$OUT/build_e_full.log"

cp -a samples/build-npu "$OUT/samples_build_npu" 2>/dev/null || true
cp -a examples/build "$OUT/examples_build" 2>/dev/null || true
mkdir -p "$OUT/op_kernel_logs"
find build/op_kernel -name '*.log' 2>/dev/null | head -200 | while read f; do
  cp -a --parents "$f" "$OUT/op_kernel_logs/" 2>/dev/null || true
done

cd "$LOG_ROOT"
tar -czvf "${EXPNAME}_logs.tar.gz" "${EXPNAME}_logs"
ls -lh "${EXPNAME}_logs.tar.gz"
echo ">>> Download: ${LOG_ROOT}/${EXPNAME}_logs.tar.gz"
```

### E2. 推荐：无密码归档（云端不 push）

CodeArts **不要** `git push`（会要账号/密码）。只做 E1 打好 tar 后：

1. 资源管理器里对 `/mnt/workspace/${EXPNAME}_logs.tar.gz` **右键 → Download**（无密码）
2. 本机放到例如 `Downloads\`，告诉 Agent 路径
3. **本机**解压进仓并 push（本机已配过凭据时通常也不再问密码）：

```powershell
cd "D:\sageAttention2 算子开发任务书"
$exp = "EXPYYYYMMDD_HHMMSS"   # 与 tar 名一致
$tar = "$env:USERPROFILE\Downloads\${exp}_logs.tar.gz"
$dir = "tasks\active\PReluGradUpdate_2329\explorations\logs\$exp"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
tar -xzf $tar -C $dir --strip-components=1
# 只保留 .log / SOURCE；大 bin 可删后再提交
git add "tasks/active/PReluGradUpdate_2329/explorations/logs/$exp"
git commit -m "PReluGradUpdate: archive $exp cloud logs"
git -c http.sslVerify=false push origin main
```

### E3. 备选：云端直接 push（需要已存凭据 / PAT，会要密则改用 E2）

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull
mkdir -p tasks/active/PReluGradUpdate_2329/explorations/logs/${EXPNAME}
# 只拷文本 log，避免整棵 build
cp -a /mnt/workspace/${EXPNAME}_logs/*.log \
  tasks/active/PReluGradUpdate_2329/explorations/logs/${EXPNAME}/ 2>/dev/null || true

git add tasks/active/PReluGradUpdate_2329/explorations/logs/${EXPNAME}
git commit -m "PReluGradUpdate: archive ${EXPNAME} cloud logs"
git push origin main
```

大 `.o` / 整棵 `build/` **不要**进 git；用 `tar.gz` 本地下载即可。

---

## 不要做的事

- 不要把整个 `build/`、`data/`、大 profiling 网页上传 / 推进 git  
- 不要只截图、不留 `.log`（纠错困难）  
- 不要在云端改完却忘了 `git pull`（会覆盖）  
- 云端若改了代码：在 `cann-ascendc-ops-collab` 里改 → commit → push，再回本机 `git pull`
