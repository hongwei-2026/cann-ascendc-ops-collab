# CodeArts 同步（无 IDE 远程时用 git，勿网页狂传）

仓库：https://github.com/hongwei-2026/cann-ascendc-ops-collab  
本题实现目录：`tasks/active/PReluGradUpdate_2329/workdir/code/`

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
rsync -a --delete \
  cann-ascendc-ops-collab/tasks/active/PReluGradUpdate_2329/workdir/code/ \
  prelu2329/code/

# 4) 进去构建（按题包 README；环境先 source cann）
cd prelu2329/code
# source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 路径以云环境为准
bash build.sh -j4
# DEVICE_ID=0 bash build.sh -j4 -e
```

若无 `rsync`：

```bash
rm -rf /mnt/workspace/prelu2329/code
cp -a /mnt/workspace/cann-ascendc-ops-collab/tasks/active/PReluGradUpdate_2329/workdir/code \
      /mnt/workspace/prelu2329/code
```

---

## C. CodeArts — 以后每次同步（本机已 push）

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull origin main

rsync -a --delete \
  /mnt/workspace/cann-ascendc-ops-collab/tasks/active/PReluGradUpdate_2329/workdir/code/ \
  /mnt/workspace/prelu2329/code/

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

## 不要做的事

- 不要把整个 `build/`、`data/`、大 log zip 网页上传  
- 不要在云端改完却忘了 `git pull`（会覆盖）  
- 云端若改了代码：在 `cann-ascendc-ops-collab` 里改 → commit → push，再回本机 `git pull`
