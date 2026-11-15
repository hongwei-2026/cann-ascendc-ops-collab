# 平台粘贴用本地超链接（Q006 · tip `4db2b31`）

2026-11-15：云测 `-e` **PASS**（opc×4；`var_out.bin` mismatched=0/64）。`profile_q005` narrow/wide_in = COMPLETE（medians 待从云端 `/tmp/qus2330_q006_rows` 补档）。

**现行采纳基线仍是 Q002**，直到平台可见点相对 Q002 有 ≥2% 收益。  
对照 Q002：`5.33 / 5.74 / 6.52 / 5.67 / 7.83` µs。增益 &lt;2% 或精度挂 → 回退 `people/.../explorations/Q006/baseline/`。

Kernel header LF SHA256（本机复核）：`d3eeea4024f4f67a17d683cf04693b3b2d6b580ca26cf09f7cae6fbbf26f303d`

## Host

- [quant_update_scatter_def.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/op_host/quant_update_scatter_def.cpp)
- [quant_update_scatter_infershape.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/op_host/quant_update_scatter_infershape.cpp)
- [quant_update_scatter_tiling.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/op_host/quant_update_scatter_tiling.cpp)

## Kernel

- [quant_update_scatter.cpp](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/op_kernel/quant_update_scatter.cpp)
- [quant_update_scatter.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/op_kernel/quant_update_scatter.h)
- [quant_update_scatter_tiling_data.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/op_kernel/quant_update_scatter_tiling_data.h)
- [quant_update_scatter_tiling_key.h](file:///D:/sageAttention2%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E4%B9%A6/people/yuhongming-2026/QuantUpdateScatter_2330/workdir/code/op_kernel/quant_update_scatter_tiling_key.h)
