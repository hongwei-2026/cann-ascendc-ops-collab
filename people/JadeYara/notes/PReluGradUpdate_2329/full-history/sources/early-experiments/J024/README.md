

## J024 — S=1 split-C / batch 分核，通道块权重驻留

目标：仅覆盖J018不能完整C驻留的C>maxTile路径。基于J023，新增split-C模板：C分成ceil(C/maxTile)个平衡的64元素对齐块，每通道块分hardwareCoreNum/cTiles组batch，组内遍历batch并持有权重队列tensor；沿用现有CopyIn/Compute/CopyOut数学与同步。仅S=1向量权重、C按32B对齐、cTiles<=硬件核数且每batch组至少2行时启用，保持近满核和权重摊销。tile预算不增加，旧路径保留。先498原精度及新大C/分核/尾块边界三dtype，再J018/J023/J024同机冷热和PipeUtilization对照；上游只借鉴调度概念，许可与commit见J021文档，不直接移植950/arch35实现。


实测已完成，见 [RESULTS.md](RESULTS.md)。保留为专项候选：大 batch/C=8192 有稳定收益，C=32768 部分形状回退。
