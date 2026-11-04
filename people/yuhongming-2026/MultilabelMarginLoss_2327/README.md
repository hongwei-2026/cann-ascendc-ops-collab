# yuhongming-2026 · MultilabelMarginLoss_2327

- **云端同步用这一版**：`workdir/code/`
- tip：同步 hongwei DMA 写回 + ScanPrefix 去 const + classTile（相对旧标量 baseline）
- 本地闸：EXP_M003 **PASSED**（910B4 · `y`/`is_target` 0 mismatch）
- 共享台账 / 设计 / 材料：`tasks/active/MultilabelMarginLoss_2327/`

```bash
python3 workdir/code/tests/local_semantic_check.py
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd workdir/code && DEVICE_ID=0 bash build.sh -j4 -e
```
