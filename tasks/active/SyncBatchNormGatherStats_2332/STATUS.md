# 状态 — SyncBatchNormGatherStats_2332

| 项 | 内容 |
|----|------|
| 状态 | **`实现中`（S006 直接除法精度实验，待 910B 回归）** |
| Owner / 座位 | yuhongming-2026 |
| 更新日期 | 2026-11-19 |
| tip | **`b2271b4`**（S006：`accSum/accSq` 改向量 `Div`） |
| 云测 | S006 **待** `CLOUD-S006.md`（`-e` + `tests/s006_rounding.py`） |
| 平台 | S002/S003/S004 均为 **WA**（1–5 Pass；隐藏失败） |
| 提交 | 910B 验证后按 `deliverables/SUBMIT_LINKS_S006.md` 重交 |
| 下一闸 | `CLOUD-S006.md`：先云冒烟，再跑固定舍入边界测试 |

## 一句话

S006 的 CPU 数值模型发现有限值舍入分歧；设备和平台结果尚未知，不宣称 WA 已修复。
