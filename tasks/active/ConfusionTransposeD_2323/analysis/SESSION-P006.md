# P006 会话卡片

2026-11-11；ConfusionTransposeD_2323；hongwei-2026；Implementer。

唯一目标：为P004的span=1标量回退增加步长DMA路径，用精确回归和同机profile判断性能收益，不宣称平台RE已修复。

仓库：`D:/华为算子比赛/ConfusionTransposeD_2323_perf_work`，main；只写本题tasks与`people/hongwei-2026/ConfusionTransposeD_2323/`，只读其它目录。已读AGENTS、05协议、02门禁、CLAIM、STATUS、ledger、SUMMARY及DEC004；应用已读CANN开发流程。无子Agent/并行实现。

Owner“继续优化”授权在平台RE未定位、无同机TBE的情况下继续有界候选开发，不视为验收豁免。基线P004=4761239；P005三类Host拒绝保持当前边界，不因优化改报PASS。新路线在ledger先登记，准备/实现/CPU验证≤90分钟，云端回传修复≤30分钟，等待Owner执行不计主动用时；到期或证据不足按DEC止损。

技术假设：将最后非单元输出轴转换成GM规则stride，MTE2将多条单元素块放入32B补齐的UB槽，再由MTE3把槽密集写回GM；srcStride单位与UB补齐须按910B DataCopyPad语义核对。控制blockCount≤4095、srcStride≤uint32、核边界与输出tile对齐32B；只对可证明的布局分派，否则保留已测路径。

验证：CPU独立NumPy对照覆盖轴排列/任意reshape顺序、每字节宽/尾块、拷贝计划地址及UB上界；云端样例build -e、原22项bitwise、新probe bitwise；P004/候选同机msprof测中位数并保留原CSV与错误日志。用户手动CodeArts运行，无直接SSH，准备一键脚本和终端可复制的紧凑证据，禁止全局DEBUG洪水。未经云测不称NPU通过或平台可提交。
