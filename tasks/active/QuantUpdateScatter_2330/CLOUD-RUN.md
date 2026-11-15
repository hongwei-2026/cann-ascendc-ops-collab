# 云端测试命令 — QuantUpdateScatter_2330

机器示意：`developer@...:/mnt/workspace`（截图环境）。

## 0. 拉代码

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git fetch origin
git checkout main
git pull --ff-only origin main
```

代码目录：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab/people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code
```

## 1. 环境

按云镜像实际路径加载 CANN（二选一或按现场改）：

```bash
# 常见
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 或
source /usr/local/Ascend/cann/set_env.sh
```

可选：

```bash
export DEVICE_ID=0
```

## 2. 功能冒烟（必做）

```bash
bash build.sh --make_clean
bash build.sh -j8 -e
```

期望：样例跑通、输出与 golden / reference 一致。  
把终端全文保存：

```bash
mkdir -p /tmp/qus2330_q001
bash build.sh -j8 -e 2>&1 | tee /tmp/qus2330_q001/build_e.log
```

回传到协作仓（在仓根执行）：

```bash
mkdir -p tasks/active/QuantUpdateScatter_2330/explorations/logs/EXP_Q001
cp /tmp/qus2330_q001/build_e.log tasks/active/QuantUpdateScatter_2330/explorations/logs/EXP_Q001/
# 然后 git add / commit / push（或把 log 下载到本地再推）
```

## 3. 可选 UT

```bash
# 需本机 googletest + nlohmann/json，路径按现场改
NLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include \
GTEST_SOURCE_DIR=/path/to/googletest \
bash build.sh -j8 -u
```

## 4. 性能（有门禁 shape 后再跑）

先确认题目门禁 case；不要用 samples 小 shape 下结论。

```bash
# 示例：msprof（命令以云镜像文档为准）
# msprof --application="bash build.sh -e" --output=/tmp/qus2330_prof
```

结果摘要写回 `path-ledger.md` 的 Q001/Q002 行。

## 5. 失败时

- 编译错误 / WA：保留**全文** log，不要只截图。  
- 精度挂：优先怀疑 RINT 路径被改、或 SyncAll/拷贝竞态。  
- 回滚单文件：`git checkout -- people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/op_kernel/quant_update_scatter.h`

## Q005 独立采集入口（本轮仅准备，未执行）

现行生产基线为 Q002。先完成并归档 bash build.sh -j8 -e，再在本题 code 目录执行：

    bash examples/profile_q005.sh /tmp/qus2330_q005_run01

只采同址/异址拷贝配对可追加 --case-id copy_heavy_out --case-id copy_heavy_in。须使用新输出目录。完整说明见 people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/tools/README_Q005.md。

默认每例 10 次预热+50 次重复、3 轮；保存完整采集日志、原始 profiler 数据、源码/包/程序哈希和逐轮精度结果。26 组为人工诊断矩阵，不是官方 50 点。RAW_ONLY/INCOMPLETE 不能当作性能通过，采集后归档本题 EXP_Q005，先分析证据再新开道路。
