# J035 与公开实现的源码对照

固定公开快照：`e62693c8bd5946eebcf2ffaacb37d2bcb892bbbf`。2026-11-05 本次独立 clone 的 main 仍是此提交。原本地工作仓 HEAD 为 `fc85a48701b1fb578cd8e5208c9b196a471ba267`，落后 51 个提交，所以比较依据采用固定公开快照，个人 J 系列采用本地冻结版本。

| 实现 | 实际结构 | 与 J035 的关键区别 |
|---|---|---|
| hongwei P105 候选 | 扁平全核切分；每 tile 标量权重 Muls；准备权重前移至 DeQue 前；四个 depth=1 队列 | 每通道 tile 受 innerSize 限制；没有 J035 的 S1/merged/resident/split-C 专用类；P105 只有已绑定日志的 FP32 小样例，平台完整收益未知 |
| yuhongming P032 系列 | 整张量能放 UB 则单 tile；整样本能放则按样本切分；否则扁平；FP32 权重展开复用 | 大样本每 tile 重建权重；29/37 字节预算，旧 CPU 单测仍有 21/29 常数；未取得独立绑定源码的完整官方性能 |
| JadeYara J035 | normal、S1、merged、resident、split-C 五类编译期路径；共享权重按规模选核；固定 C 块跨 batch 复用；UB 相位布局 | 低精度权重只搬一次但仍每 tile Cast；当前进一步缓存需要验证额外缓冲代价 |

各版本都使用 Compare/Select，低精度有 FP32 中间量；BF16 标量位操作辅助不能与 910C 向量 Cast 混为一谈。hongwei 历史已采用 P103 与最新 P105 候选需分别讨论，不能把 P105 写成官方最佳。

| 文件级比较 | 相同 | 修改 | 左独有 | 右独有 |
|---|---:|---:|---:|---:|
| J035 / hongwei | 35 | 6 | 5 | 1 |
| J035 / yuhongming | 31 | 9 | 6 | 0 |
| hongwei / yuhongming | 32 | 8 | 2 | 0 |

完整差异：J035/hongwei（本地留存）、J035/yuhongming（本地留存）、两位作者（本地留存）、文件比较清单（本地留存）。保留原作者来源，不把其他作者目录当作 JadeYara 代码。

## 历史教训的出处

- [DEC001/P102](../../../../tasks/active/PReluGradUpdate_2329/decisions/DEC001-p102-rollback.md)：对齐搬运回退；需要同 shape/dtype/硬件、指纹、交替对照。
- [DEC002/P104](../../../../tasks/active/PReluGradUpdate_2329/decisions/DEC002-p104-platform-regression.md)：常驻零缓冲减少操作，前三点仍慢。结果形态像 J048，但根因未证实相同。
- [EXP025/P105](../../../../tasks/active/PReluGradUpdate_2329/explorations/EXP025-early-weight-preparation.md)：只前移权重准备，尚无完整平台收益；编译三个 key 不等于三个 dtype 全精度通过。
- [EXP018](../../../../tasks/active/PReluGradUpdate_2329/explorations/EXP018-case4-unaligned-weight-ub.md)：Duplicate 的 UB 对齐问题。
- [EXP020](../../../../tasks/active/PReluGradUpdate_2329/explorations/EXP020-review-p032-vs-p100.md)：核对真实源码，不照旧 FIX_NOTES 推断当前实现。

这些相对链接方便浏览当前仓库；历史判断绑定上述固定提交。旧文档、测试常数与源码不一致时，以绑定版本的源码及原始设备结果为先。

三个完整 diff 及文件清单保留在本地 `handoff_pinned_e626/`，本次公开同步只包含上述比较说明。
