# 云端测试命令 — SyncBatchNormGatherStats_2332

机器示意：`developer@...:/mnt/workspace`。  
本轮道路：**S002** / 日志目录：**EXP_S002**。

## 0. 拉代码

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git fetch origin
git checkout main
git pull --ff-only origin main
```

代码目录：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code
```

## 1. 环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 若上面路径不存在，改用：
# source /usr/local/Ascend/cann/set_env.sh

export DEVICE_ID=0
```

## 2. 功能冒烟（本轮必做）

```bash
mkdir -p /tmp/sbngs2332_s002
bash build.sh --make_clean
bash build.sh -j8 -e 2>&1 | tee /tmp/sbngs2332_s002/build_e.log
```

期望：样例跑通、输出与 golden / `samples/reference.py` 一致。  
样例 PASS ≠ 平台 50 点 PASS。

## 3. 回传全文 log（仓根执行）

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
mkdir -p tasks/active/SyncBatchNormGatherStats_2332/explorations/logs/EXP_S002
cp /tmp/sbngs2332_s002/build_e.log \
  tasks/active/SyncBatchNormGatherStats_2332/explorations/logs/EXP_S002/
# 然后本地或云上 git add / commit / push（把全文 log 推到 main）
```

## 4. 可选 UT（本轮可不跑）

```bash
NLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include \
GTEST_SOURCE_DIR=/path/to/googletest \
bash build.sh -j8 -u
```

## 5. 失败时

- 全文 log 必须归档；禁止只留截图。
- 回传后更新本题 `path-ledger` + `dev-log`。
