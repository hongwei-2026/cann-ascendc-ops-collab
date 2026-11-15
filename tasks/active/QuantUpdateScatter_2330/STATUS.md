# 状态 — QuantUpdateScatter_2330

| 项 | 内容 |
|----|------|
| 状态 | **`性能中`**（ZIP 功能基线已导入；Q001 有界优化待云测） |
| Owner | yuhongming-2026 |
| 更新日期 | 2026-11-15 |
| 代码座位 | `people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/` |
| 合入仓 | cann/ops-nn（待确认正式 MR 路径） |
| 阻塞 | 缺同机 TBE 对照表与门禁 shape；本机无 NPU |
| 下一闸 | 云端 `bash build.sh -j4 -e` 精度冒烟 → 再 profile 大 shape |

## 一句话

从 `QuantUpdateScatter_2330.zip` 导入正确性基线，落地 Q001（多核拷贝 / indices 批量 / scales·zp 复用）。接力读 `AI-HANDOFF.md`。
