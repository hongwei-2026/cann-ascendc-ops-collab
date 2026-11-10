# hongwei-2026 · ConfusionTransposeD_2323

从 `people/yuhongming-2026/ConfusionTransposeD_2323/` 拷贝座位（EXP_C001 样例 PASS）。本座位先做 **扩泛化 / 云测**，不抢改 yuhongming workdir。

## 路径

```text
people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/
```

## 本地 / 云测

```bash
source /path/to/cann/set_env.sh
cd people/hongwei-2026/ConfusionTransposeD_2323/workdir/code
python3 -B tests/local_semantic_check.py          # CPU，无 NPU
DEVICE_ID=0 bash build.sh -j4 -e                  # 原样例
DEVICE_ID=0 bash tests/run_c002_cloud.sh          # 样例 + 泛化回归，日志打 tar
```

台账：`tasks/active/ConfusionTransposeD_2323/`（P002 / EXP_C002）。
