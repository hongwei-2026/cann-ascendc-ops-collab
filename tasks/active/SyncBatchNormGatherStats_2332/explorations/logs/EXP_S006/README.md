# EXP_S006 logs

云 `-e` + `s006_rounding.py`：**PASS**（见 `RESULT.md` / `SOURCE.json`）。

云机若已 `tee` 全文，可覆盖本目录后 push：

```bash
# 在云仓根，且 S006_LOG 已指向本目录时：
cd /mnt/workspace/cann-ascendc-ops-collab
git add tasks/active/SyncBatchNormGatherStats_2332/explorations/logs/EXP_S006/
git commit -m "SyncBatchNormGatherStats_2332: archive EXP_S006 cloud PASS logs."
git push origin main
```
