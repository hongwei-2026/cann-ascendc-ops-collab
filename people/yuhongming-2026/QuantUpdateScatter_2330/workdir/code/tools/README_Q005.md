# Q005 — Q002 瓶颈采集工具

本轮只补测量能力，生产 host / kernel 保持 2d3c61c 中的 Q002。不是加速候选；没有设备实测收益。

## 本地检查

在本题 code 目录执行（或使用脚本绝对路径）：

    python -B tests/local/run_q005_support.py --cxx /path/to/g++

该检查会编译实际诊断 runner，使用同步 CPU ACL stub 调用 26 组诊断输入、每组 3 次，并逐字节对齐 reference。另检查错误参数、输入哈希、损坏输出、预热剔除和不完整导出。它不验证 CANN 编译、设备流水或性能。

## 设备采集入口（本轮未执行）

先加载 CANN 环境，在本题 code 目录完成 bash build.sh -j8 -e 并保存完整日志。确认使用 Q002 构建产物及正确的自定义算子路径。然后执行：

    bash examples/profile_q005.sh /tmp/qus2330_q005_run01

必须使用新目录，已有证据不覆盖。默认 10 次预热 + 50 次采样、3 轮；每次执行重新获取 executor、同步完成后才复用 workspace。每轮交替 case 顺序。可先采一个成对场景：

    bash examples/profile_q005.sh /tmp/qus2330_q005_copy01 --case-id copy_heavy_out --case-id copy_heavy_in

26 组是人工诊断矩阵：tiny / copy_heavy / narrow_rows / wide_rows / many_batches 的同址/异址配对，以及四 dtype 签名、可选 zp 和 reciprocal。成对场景的输入与 golden 哈希一致。它们不代表官方 50 点，不能替代隐藏 45 点回归或同机标杆。

自动模式优先使用镜像支持的 msprof --application=... --aic-metrics=PipeUtilization，否则检查 msprof op；也可追加 --profiler-mode op 显式选取。保留两套 --help 和实际调用。不会在采集中编译，也不会同时传 --application 和 --export。镜像选项不支持时保留错误日志并停止，不盲试未知参数。

## 证据与口径

- 采集目录保存 runner 构建日志、源码/包/程序哈希、case 参数、原始 profiler 目录、完整 collector.log、每轮输出及逐字节校验结果。
- COMPLETE：找到有时间戳、正时长且任务数量恰好匹配的 op_summary，按时间顺序剔除预热，输出任务时长的 median/p10/p90。任务时长不是纯 vector 计算时间。
- RAW_ONLY：有非空 PipeUtilization / OpBasicInfo 原始表；不猜逐核行和 launch 的对应关系，不生成时长聚合。需要人工分析原表/时间线。
- INCOMPLETE：缺少导出或不能安全聚合。缺失值不会填 0。执行校验通过和 profile 完整度分别记录；退出 0 不等于性能门禁通过。
- run.json 中的包哈希和源码哈希只能标识本地文件，不能单独证明设备加载的代码；仍需保留构建 -e 日志和自定义算子路径。

已有采集目录可以本地汇总：

    python -B tools/profile_q005.py summarize --directory /path/to/case/prof --warmup 10 --repeat 50

拿到证据后，先比较同址/异址差值，再看 MTE2、MTE3、Vector 和 Scalar 指标及时间线。利用率不能直接当作阻塞比例；没有流水证据不宣称“真双缓冲”收益。先写下一条新假设与台账，再实施生产改动。

## 本轮实测边界

26 组 × 3 次 actual-runner CPU 接口检查通过；四签名注册/dispatch/错误信息契约检查通过；采集 shell 语法检查通过。旧 run_q002_kernel_model.py 因当前主仓未包含 explorations/Q002/baseline 而未能运行，保留失败日志，未补造 PASS。生产 host/kernel 对 HEAD 的 diff 为零。

CANN 编译、NPU 精度、msprof 实采、同机标杆与性能收益均 NOT_RUN。用户要求仅本地，本轮未 commit、push 或运行云端命令。
