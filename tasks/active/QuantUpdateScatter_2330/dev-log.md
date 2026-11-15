# Dev Log — QuantUpdateScatter_2330

## 2026-11-15

- 从 ZIP 导入代码到 `people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/`（排除 build）。
- 落地 **Q001**：多核 `var→out` 拷贝、indices 批量进 UB、scales/zp 按 lastDim tile 复用；tiling 按 `totalNum` 抬核。
- 写 `AI-HANDOFF.md` + `CLOUD-RUN.md` 供云测与其他 AI 接力。
- 本机无 NPU，未跑 `-e`；不得把精度写成 PASS。
