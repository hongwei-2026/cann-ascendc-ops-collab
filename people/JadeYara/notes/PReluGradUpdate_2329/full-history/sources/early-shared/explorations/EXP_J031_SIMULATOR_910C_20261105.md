# J031：现有模拟器成功复现 J026/J029 依赖

本轮只做仿真取证，未修改或重新编译算子，也未执行新实板性能作业。J026 仍是基线，J027 仍是条件候选，J029 不因仿真结果晋级。

## 环境与入口

本机 CANN 9.0.0 的 `msprof op simulator --help` 可用。`tools/simulator/Ascend910_9362` 明确映射到 `dav_2201`，release_config.json 的 CA 模式列出的五个库全部存在，已记录 SHA256：libpem_davinci、libnpu_drv_camodel、libstars、libmodel_top、libruntime_camodel。没有安装或更新组件。

已核现有入口为 C++ ACLNN `J021-J026-build/device_probe` / `J021-J029-build/device_probe`，并非自建 kernel 直调程序。安装包自带 kernel-launcher，但不接受 `--help`，该探测退出 1；没有继续猜参数。官方文档也提供 `.o + JSON --config` 入口，不过本轮 ACLNN 单次调用已成功，因此不需要另建入口、手工构造 tiling 或重编译。

准确命令记录在各案例 execution.json：

```text
msprof op simulator --soc-version=Ascend910_9362
  --aic-metrics=PipeUtilization,ResourceConflictRatio
  --launch-count=1 --core-id=0|2 --timeout=3 --dump=on
  --output=<独立目录> <原版device_probe> <单行plan> reuse 0
```

参数通过 subprocess 参数数组传递，`0|2` 不经过 shell 管道解释。plan 为 FP16 `[127,8192,1]`、weights=8192、iterations=1、gap=0、refresh=0。输入沿用 J024/J030 已归档案例。独立进程环境选择各版本已有 custom OPP，不改持久环境。

两次仿真均退出 0：J026 日志开始 2026-11-05 04:00:39，解析结束 04:01:08；J029 04:01:08–04:01:36（云端日志时间）。两个输出均与原 golden 数值逐元素完全一致，且两版本输出文件 SHA256 一致。这里只覆盖这一个 FP16 案例，三 dtype 泛化仍引用已有实板记录。

## 保留原 tiling 的证据

模拟器初始化显示 24 个 AIC、每 AIC 两个 Vector，但实际指令 dump 中运行 block ID 为 0–39，不能把模型最大拓扑当作本算子用了 48 核。tiling 读取值包含 totalNum=0xfe000、tileLen=0x1000、channelNum=0x2000、splitBatchGroups=0x14，即 1,040,384 元素、tile4096、C8192、20 batch 组；与实板目标相符。

注意工具重命名：解析目录 `simulator/core2.veccore0` 对应逻辑 block2，原始 dump 是 `dump/core1.veccore0.*`；原始 `dump/core2.veccore0.*` 的 BLOCKID 则为4。分析脚本通过 `SPR:BLOCKID` 匹配，未按同名目录错误对齐。block0 做6个 batch，block2 做7个 batch。

原 runner 和三 dtype 包内 `.o` 已重新 SHA 校验，与 J030 provenance 全部一致。完整 raw dump、PC、指令参数、解析 trace/CSV 已保存。

## 实际观察

下表“时间”仅为模拟器报告的单核 duration，不是 NPU Task Duration；本轮每版本仅一次仿真。

| 逻辑 block | batch 数 | J026 模拟 duration/μs | J029 模拟 duration/μs | 原始指令数 J026→J029 |
|---|---:|---:|---:|---:|
| 0 | 6 | 8.61 | 8.26 | 4236→3325 |
| 2 | 7 | 9.54 | 9.17 | 4819→3767 |

block2 的 MTE2 数据拷贝仍为15条（每批 grads/features 两条，加首次 weight 一条），MTE3 仍为14条（每批两输出）。合并队列减少了管理指令，没有减少数据拷贝次数。Vector 指令总数447→419，包括事件/同步；不能将减少数全部称作计算指令。

关键依赖已落实到 PC 和 UB 地址：

1. FP16 输入搬到 UB 0x0000/0x2000；转换结果写到独立 FP32 缓冲 0x13000/0x17000。
2. 后续数学计算、Compare/Select 和输出转换使用 FP32 缓冲；输入槽的 VEC→MTE2 释放标志仍在整段 Compute 末尾发出。
3. 下一批输入搬运在 MTE2 上执行 WAIT_FLAG，等待前一批的释放标志。J029 合并队列后这个依赖仍在。

block2 后续 batch 的“features 输入最后一次 Cast 完成→输入释放 SET_FLAG 完成”间隔：J026 为 **1036 模型周期**，J029 为 **1020 模型周期**。首批分别1985/1713周期，含首次路径开销。这里是输入槽生命周期中的间隔，不是可全部兑现的算子收益。

一个直接可查的后续 batch 等待实例（trace 时间值按 μs，`displayTimeUnit=ns` 仅是显示设置）：

| 版本 | MTE2 WAIT PC | WAIT开始→结束 | 时长 | 对应 VEC SET_FLAG PC / 时间 |
|---|---|---|---:|---|
| J026 | 0x11b0d5d4 | 9.553514→10.052973 | 0.499459 | 0x11b0e648 / 10.051892 |
| J029 | 0x11b0d570 | 9.098378→9.751892 | 0.653513 | 0x11b0e36c / 9.750811 |

表中 SET_FLAG 时间是事件起点。该标志之后 WAIT 解除，随后才能复用输入槽。完整逐批原始 tick、PC 和解析事件见 J031-analysis.json。raw popped→completed 的驻留周期与解析 trace 时长不是同一个字段；CSV 中 BAR/WAIT 累计时间也不能相加当独占开销。没有从这些数据推导可直接节省数微秒。

## 与实板结果的关系和剩余缺口

仿真 J029 单核 duration 比 J026 少约0.35–0.37μs，但 J029 历史4轮实板性能协议为0/15组合获得四轮一致收益；J030 的冷/热 replay 下结果也有方向变化。模拟器没有复现全部缓存、频率、内存系统条件，**不据此推翻实板结论**。

此次找到了输入队列复用依赖的具体指令、缓冲地址和先后顺序；仍没有 910C 实板核内指令时序，也未把 J030 的 VECTOR IQ full、UB bankgroup 冲突精确归因到某个源码行或单一临界路径。默认二进制缺 `.debug_line`，工具明确警告且 code_exe.csv 为空；PC/指令可用，源码行热点不可用。ResourceConflictRatio 选项被接受，但本轮产物没有独立硬件式冲突计数表，不声称获得新的实板 bank 冲突归因。

## 最小后续实验

以 J026 为基线，仅在 FP16/BF16 独立 FP32 转换路径中，将输入队列 FreeTensor 移到两次输入 Cast 与必要 PIPE_V 同步之后。保持 tile、核数、输入槽数、全部数学表达式与输出队列不变。FP32 使用输入别名，不能同样提前释放。先检查生成指令中 VEC→MTE2 SET_FLAG 是否前移、后续读取是否仍指向输入槽，再做三 dtype 精度（含边界及已有留出输入）与同机交替实板 A/B。该方案尚未实施，不承诺收益；若标志前移仍不改善 Task，停止此方向。

这与 J028 的增加输入双缓冲/预取调度不同：不增加 UB 槽位，而是缩短已完成最后读取的输入槽生命周期。它能直接检验本轮观察到的依赖，避免继续扫描 tile 或核数。

## 归档与完整性

`J031-simulator-evidence.zip`：30,703,747字节，621个原始文件。

SHA256：`7241fc803272ed618a4c40bf5625aaeefe3e642ff5214922af9cf48da9528460`。

已取回并验证 SHA256、ZIP CRC、manifest中每文件哈希。该归档包含完整两次仿真原始数据和输出；复原原代码、输入、golden、runner/包需要已有自包含 `J030-910C-evidence.zip`，SHA256 `03dd974b056058e9a7d8e2268f5de9dd149b5bbfe80dfecc9adcf913cd641536`，本地依赖也已验证。下载首块一次连接中断，重取该完整块后校验成功，未重复仿真。最终无在途 msopprof/device_probe 作业。

首次库存脚本误读 release_config.json 层级导致只读查询中断；改为正确 conf 目录完成查询。失败原日志一并保留。没有权限拒绝、安装、网络改动、竞赛提交或 push。

官方依据（固定 commit，实际参数以本机 help 为准）：

- [模拟器使用指南](https://github.com/Ascend/msopprof/blob/16beabbbb76f12136cc940866990b7f3e0289602/docs/en/user_guide/msopprof_simulator_user_guide.md)
- [JSON直调配置](https://github.com/Ascend/msopprof/blob/16beabbbb76f12136cc940866990b7f3e0289602/docs/en/user_guide/extended_functions.md)
- [模拟器数据字段](https://github.com/Ascend/msopprof/blob/16beabbbb76f12136cc940866990b7f3e0289602/docs/en/user_guide/msopprof_simulator_performance_data.md)
