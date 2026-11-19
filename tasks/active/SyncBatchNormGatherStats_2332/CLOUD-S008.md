# S008：大 C 多 tile 同步验证

先保留 S007 原始日志和失败用例数据；不要删除 `samples/build/s007/`。
本方案是同步假设，尚未证明为隐藏 WA 根因；PROBE 也不是内置算子对照。

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull --ff-only origin main
git log -1 --oneline
export S008_REPO="$PWD"
export S008_LOG="$S008_REPO/tasks/active/SyncBatchNormGatherStats_2332/explorations/logs/EXP_S008"
export DEVICE_ID=0
mkdir -p "$S008_LOG"
cd "$S008_REPO/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code"
set -o pipefail
python3 -B tests/s007_matrix.py --self-test 2>&1 | tee "$S008_LOG/selftest.log"
bash build.sh -j8 -e 2>&1 | tee "$S008_LOG/build_e.log"
```

构建及样例通过后，使用与 S007 完全相同的用例编号及随机种子独立运行三次；
每次输出目录独立，不覆盖原始证据：

```bash
for run in 1 2 3; do
  python3 -B tests/s007_matrix.py --case large_f32_i32_c524289 \
    --output "samples/build/s008-run$run" 2>&1 | tee "$S008_LOG/large-$run.log"
done
```

三次均通过后再跑完整回归：

```bash
python3 -B tests/s007_matrix.py --output samples/build/s008-matrix 2>&1 | tee "$S008_LOG/matrix.log"
cp samples/build/s008-matrix/results.json "$S008_LOG/results.json"
```

记录四个输出各自的 mismatch 数量；即使 batch_mean 转为 0 错，也要检查
S007 曾出现的 batch_invstd 7 处和 new_variance 13 处差异。失败或构建错误同样保留全文。
严格矩阵仍有余差时，不要声称全部通过。S008 云端实测结果和样例比例规则分析见
`explorations/EXP_S008.md` / `EXP_S009.md`；平台判定规则仍需实际提交确认。

请将 S007 的 selftest.log、build_e.log、matrix.log、results.json 和 S008 的
全部日志推送到各自的 `explorations/logs/EXP_S007/`、`EXP_S008/`；
`samples/build/` 的大二进制数据单独保存，不要提交进 Git。
