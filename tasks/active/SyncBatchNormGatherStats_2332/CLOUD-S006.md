# S006 云端精度验证

在已配置 CANN 的 910B 云机运行；日志保存在本题台账内。若任何步骤失败，保留日志，停止后续平台提交。

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
git pull --ff-only origin main
git log -1 --oneline

export DEVICE_ID=0
export S006_REPO="$PWD"
export S006_LOG="$S006_REPO/tasks/active/SyncBatchNormGatherStats_2332/explorations/logs/EXP_S006"
mkdir -p "$S006_LOG"
cd "$S006_REPO/people/yuhongming-2026/SyncBatchNormGatherStats_2332/workdir/code"
set -o pipefail
bash build.sh -j8 -e 2>&1 | tee "$S006_LOG/build_e.log"
```

构建及样例通过后，运行固定舍入边界：

```bash
python3 tests/s006_rounding.py 2>&1 | tee "$S006_LOG/rounding.log"
```

脚本逐项打印四输出 mismatch；输入、reference 和设备输出保存在 `samples/build-s006/`。两项都通过后，按 `deliverables/SUBMIT_LINKS_S006.md` 上传 7 个源码文件；两个测试通过不代表平台 50 点通过。请将两个日志全文归档并 push main，失败也保留。
