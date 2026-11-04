# EXP024 — P104 每核常驻零缓冲（不采纳，已撤回）

- 日期：2026-11-04
- 范围：`people/hongwei-2026/PReluGradUpdate_2329/workdir/code/` 与本题台账
- 仓库：`D:\sageAttention2 算子开发任务书`；分支 `main`；席位 hongwei-2026；不改其他题目/席位
- 已测基线：P103 `47db93e`，LF Kernel SHA256 `2a340f612ec506a6e6ae920fd8f6acd14999f28cb2d89d386d2a075ff6fb3c89`
- 候选 Kernel LF SHA256：`2c5f09e1901a05e81e345ab797d1e62d414cb898acbaa58eda39d60c4e5fcb2a`
- 状态：云测 0/192 PASS；平台 19:59 **变慢不采纳**（见 DEC002）；默认回 P103

## 新日志与目标

`logs/EXP_P103_20261103_163813/SOURCE.txt` 的 source/build 指纹一致，`build_e_full.log` 三 key Done，dx/update 均 0/192，样例 PASSED。包内没有可用于定位瓶颈的设备流水线 profile。

归档截图 `screenshots/EXP023-platform-20261103-164642.png` 的分数是 42.89，耗时 7.36/7.30/7.24/5.83/6.07 μs。42.89 比 P100 的 42.41 高 0.48 分，比 P102 的 42.06 高 0.83 分；用户所说 0.68 未在已找到的归档里对应到具体截图，不将其解释为百分比或耗时收益。截图本身无源码 hash。P103 相对 P100 耗时有升有降，不能宣称已突破。

## 假设及实现

P103 的 `zeroLocal` 与 `dxFloat` 别名，update Select 读完零值后 dx Select 才能覆盖，因此两次 Select 中间存在 WAR 屏障；下一 tile 还要重新清零。P104 分配独立 `zeroFloatBuf_`，在 Init 按 alignedCount 清零并 PIPE_V 同步一次，后续 Compute 只读。

每 tile 第一组为 Muls、Mul、Compare：输出分别为 neg、fg、mask，输入只读，操作之间无显式张量写依赖。第一道 PIPE_V 后，两次 Select 输出 dx/update，读取不变的 mask/zero/输入/乘积，不再共享可写零缓冲；第二道 PIPE_V 保证输出完成后才能转换和入队。

P103 每 tile：6 个向量操作、4 道显式 PIPE_V；P104 每 tile：5 个操作、2 道显式 PIPE_V，另加每核一次 Duplicate/PIPE_V。单 tile 时总操作数仍为 6，总显式屏障为 3；多 tile 时才摊薄零初始化。源码计数不等于机器指令数、周期或加速比，内部临时空间与编译器行为仍需编译及真机确认。

## 内存与兼容性

新增零缓冲占 `AlignUp(tileLen,64)*4` 字节。Host 每元素预算 fp32 从 25 改为 29，低精度从 33 改为 37，保留 80% UB 预算与 Select 8K 预留；maxTile 仍向下按 64 对齐。额外缓冲会缩小最大 tile，可能增加大输入的循环次数，因此不能仅凭少屏障断言性能提高。

未改变 copy API、dtype 转换、count Select、Compare 极性、channel 边界与进度、workspace、核数。没有重开 P102 对齐搬运路线。bf16 原有标量转换仍在，不能据此声称低精度大 shape 瓶颈已消除。

## 本地检查

1. `tests/compute_schedule_check.py --baseline <P103 Kernel>`：解析实际源码，对 fp32/fp16 和特殊数值对照 reference，候选 5 操作/2 屏障、P103 6/4；共享零数组跨多次调用未改写。
2. 显式 RAW/WAR/WAW 检查通过；故意移除第一道依赖屏障或将零值重新别名到 dx 时检查拒绝。初始化和末尾输出屏障、只初始化一次的断言通过。
3. `tests/local_semantic_check.py`：现有语义、channel 切块、核数、尾块检查通过；新增 UB 上界检查覆盖多种 UB 容量、fp32/低精度与对齐边界。
4. `git diff --check`：通过。

这些是 CPU 源码模型检查，不覆盖 Select 内部临时缓冲、编译器同步、真实 DMA 或设备执行，不等于 CANN/平台 PASS。

## 下一闸与止损

先确认新 Kernel 源码与 build 指纹一致，三 key 编译并运行 `bash build.sh -j4 -e`，保留完整日志；平台五点精度均通过后，与 P103 在相同硬件/输入/计时口径交替重复采时。记录 median、波动和原始数据，关注额外 UB 对大 tile 的影响。

任何编译/精度失败立即停止该候选，保留失败日志；没有稳定 ≥2% 耗时收益则不采纳，并恢复 P103。用户只授权本地修改，本轮不提交、不推送。

## Owner 复审（同日）

- **可试**：独立零缓冲消除 update→dx 的 WAR；本地/云测功能 OK。  
- **平台**：19:59 提交 1132450；8.15/8.22/8.25/5.79/5.99μs vs P103 7.36/7.30/7.24/5.83/6.07 → 前三点变差；**DEC002 不采纳**。  
- **截图**：`screenshots/EXP024-platform-20261104-195909.png`

## 本次本地执行结果

2026-11-04 19:59:09 图为 40.42 分、8.15/8.22/8.25/5.79/5.99 μs；对照 P103 历史图，前三点慢 10.73%/12.60%/13.95%，后两点快 0.69%/1.32%。P104 不采纳。

当前工作区实际恢复 P103 的 Kernel、Host tiling 和两份对应检查脚本，不再只是默认版本的文档声明；保留全部历史记录。本地语义和调度检查通过，平台回退后的耗时未复测。根因未定位，后续不再仅按减少屏障/操作数量推断提速，详见 DEC002 的本次执行段。
