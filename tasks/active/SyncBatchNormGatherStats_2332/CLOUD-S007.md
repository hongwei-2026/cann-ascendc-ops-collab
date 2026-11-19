# S007：910B 多维精度排查

S007 保持 S006 算子逻辑，仅新增测试。157 个构造用例不等于平台隐藏 45 点；
多设备与多 count 的候选公式尚无权威裁决。截图的 `*` 不能判断哪个隐藏点失败。

在已配置 CANN 的 910B 云机运行：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull --ff-only origin main
git log -1 --oneline
export S007_REPO="$PWD"
export S007_LOG="$S007_REPO/tasks/active/SyncBatchNormGatherStats_2332/explorations/logs/EXP_S007"
export DEVICE_ID=0
mkdir -p "$S007_LOG"
cd "$S007_REPO/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code"
set -o pipefail
python3 -B tests/s007_matrix.py --self-test 2>&1 | tee "$S007_LOG/selftest.log"
bash build.sh -j8 -e 2>&1 | tee "$S007_LOG/build_e.log"
```

构建和样例通过后运行；脚本会继续执行后续用例并汇总失败：

```bash
python3 -B tests/s007_matrix.py 2>&1 | tee "$S007_LOG/matrix.log"
cp samples/build/s007/results.json "$S007_LOG/results.json"
```

- 单设备、单 count 用例按参考实现比较，包含四种 dtype 组合、通道尾块、
  一维输入、不同 count 和属性、count=0/1、零/负方差及大通道。
  count=0 是探索性边界，是否属于平台合法输入仍待确认。
- `PASS` / `FAIL` 只表示与仓库 reference 的比较，NaN 与正负无穷严格区分；
  不代表平台全部通过。特殊值判定规则可能与平台不同，需要结合设备进一步裁决。
- `PROBE` 为多设备或多 count：分别列出首设备/设备求和与首 count/count 求和
  四组候选的差异；这些候选不是已证实的内置或平台真值。
- `ERROR` 表示调用或文件读取失败，不是数值正确。
- 输入、输出、参考值、候选值和单次运行日志位于
  `samples/build/s007/<用例名>/`，包括失败用例；该 build 目录不入 Git。
  保存完整云端日志，失败用例文件另行打包留存，避免把大通道数据误推入仓库。
- 将 selftest、build、matrix 三份完整日志及 results.json 归档并 push main；
  失败也保留。不要仅凭测试套件 PASS 重交平台，应先分析 FAIL/ERROR/PROBE。
