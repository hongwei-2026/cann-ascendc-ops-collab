

## J022 — 低精度 legacy S=1 的指令预取实验

J021 已复现目标小算子预热消除回退，而无关大算子预热不能消除；ICache miss 同步变化。仅在 J020 的 FP16/BF16 legacy S=1 入口增加 ICachePreLoad(4)，4×2KB=8KB，现有 AIV 16KB 上限内；其余 Host/数学/同步和三条路径不变。安装 CANN 的 dav_c220 wrapper 支持 preload，API 文档 https://asc.gitcode.com/api/SIMD-API/basic_api/cache_control/ICachePreLoad_ISASI.html 。先构建和498精度，再同机冷/热及泛化交替对照；若无重复收益则保留 J020 原版。J021 replay 独立测量，不与普通 profiler 混用。
