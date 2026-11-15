# SUMMARY — QuantUpdateScatter_2330

> 本题 ZIP 已带可运行功能基线；本 SUMMARY 压缩「性能阶段入口」，非从零三路开题全文。

## 定稿结论

1. **语义钉死**：reference.py + 内置 RINT 路径；axis=-2、quant_axis=-1。  
2. **性能主矛盾**：整表拷贝带宽 + scales/zp 重复 DMA + 串行流水。  
3. **P0 路线 = Q001**（多核拷贝 / indices UB / scales 复用）；双缓冲为 Q002。  
4. **废弃**：改量化舍入路径；无 SyncAll 的「抢跑」多核写。

## 实现优先级

| 优先级 | 项 | 状态 |
|--------|----|------|
| P0 | 功能基线导入 | 完成（P000） |
| P0 | Q001 有界优化 | 代码完成，待云测 |
| P1 | 云测 + TBE 对照 | 未做 |
| P2 | Q002 双缓冲 | 未开 |

## 风险

- 小 batch + 大 var：抬核后部分核 `batchLen=0`，只拷贝——符合设计。  
- indices > 2048/核：回退 GetValue，Q001 未彻底消灭标量读。  
- UB 增加 indices 缓冲后 ubFactor 略降，大 lastDim 需 Q003 回看。
