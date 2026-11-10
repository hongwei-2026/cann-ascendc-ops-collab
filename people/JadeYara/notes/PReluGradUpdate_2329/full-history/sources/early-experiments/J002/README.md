# J002：提交 1109131 正确性调查

2026-11-03。个人座位 JadeYara。目标：查明 WA 并补设备诊断；没有新性能优化、上传、提交或推送。原 J001 workdir、交付 ZIP 和 P103 baseline 全部保留。

## 当前结论

用户截图报告：提交 **1109131**，页面时间 **2026-11-03 22:53:30**，最终 Wrong Answer，点1显示输出错误占比100%、3.48 μs，点2–5 Skipped。本轮接受此为用户提供的状态信息，没有重新读取平台。截图的“100%”缺少逐元素日志，不能推断全部 tensor 元素都错；3.48 μs 不是有效性能成绩。

个人目录及本题既有文本记录没有此提交 ID 或对应时间，也没有上传包绑定。J001 ZIP SHA256 实算为：

`fe319504df5d033fde249f18ca6a091991c3490a21f3bc92ad0d330db6da7775`

包内42文件与当前 workdir 一致。Kernel SHA256：`2a340f612ec506a6e6ae920fd8f6acd14999f28cb2d89d386d2a075ff6fb3c89`；Host tiling：`d561da63c4a280f08a68633a6923a616ced218582d52d226075c1b2d98ef8914`。

**最小缺失证据：提交1109131实际上传的ZIP或其可信SHA256。** 文件名不能确认内容。接着需要完整编译/运行日志、失败用例 dtype/shape、两个输出首批 expected/actual。包可用 `diagnose.py identify ACTUAL.zip --known ../J001.zip` 比较；不同 ZIP 压缩元数据也会导致 ZIP 哈希不同，应再比内部源码，不能自动判定代码不同。

## 独立审计

- P103→J001 生产差分只有Host两处 `weightLen == channelNum` 改成 `weightLen > 1`。合法权重只有1或C，因此只有C=1命中变更；C>1的合法输入保持原分派。
- C=1连续tile仍用weightIndex=0；blockFactor/核心区间不变，currentNum逐次推进，64元素对齐后的buffer受maxTile限制。未发现可证实的漏写、越界或广播错误。
- Kernel、dtype key、tiling结构、数据搬运、分支和舍入与P103逐字一致。默认FP32/FP16/BF16启用，key 0/1/2与模板对应。没有发现输出顺序倒置。
- 仍需设备排查：长tile是否暴露已有多repeat/同步问题；BF16向量结果转标量读取是否由编译器正确自动同步。它们是待证假设，不能据此直接添加barrier或宣称修复。
- 当前参考是通道轴广播、两个同形逐元素输出、update不reduce。未修改golden，也不使用其他同名不同接口算子的golden。特殊值的np.where与掩码乘法差异需依真实平台合同另行验证。

**已证实的验证缺口**：原 `build.sh -e` 固定FP32 `[4,6,8]`、weights `[6]`，不经过J001改动的C=1分支。旧CPU模型不模拟设备指令或流水。J002补独立参数化设备入口，未更改生产代码。

## 诊断内容与本机结果

`device_probe.cpp` 调用现有自定义aclnn接口，参数为输入目录、dtype、shape、权重长度和device。输出先填0xA5，帮助区分未写输出；保存两个原始bin。`diagnose.py`生成30个用例（三dtype×10组），输入均为有限且乘积可精确表示的值：

- 正/负/零分支 `[1,1,65]`；
- J001命中组 `[128,1,1]`、`[32,1,17]`；
- 通道和64lane边界 `[2,3,1]`、`[2,3,65]`；
- 未改路径 `[4,6,8]`，weights `[6]`和`[1]`；
- `[600000,1,1]`，在48核、模拟256KiB UB下覆盖大tile和多tile（实际tiling须以设备为准）。

各组g循环1..8，w按0.5/2/4，x按正、负、零模式。比较器精确数值相等，符号零相等；这只是该诊断集的要求，不是平台误差阈值。它输出首8个错误的flat index/channel/expected/actual、文件长度错误和每个输出hash。

本机已运行：诊断器15项故障注入自测PASS；30组golden与存档reference对照PASS，共3,609,882个输出元素。无设备输出时检查为0/30并返回1，确认不会把缺失设备结果当成功。CPU生成的golden未写入device output目录。

**C++ runner未编译，CANN Host UT、NPU运行、P103/J001设备A/B均未执行**。没有证实生产根因，因此没有“修复版J002提交包”。

用户随后要求本地跑。已检查Windows与现有Ubuntu WSL2，并启动原本Stopped的Ubuntu进行只读检查：Linux 6.6.114.1-microsoft-standard-WSL2/x86_64，存在`/usr/bin/python3`，但未找到cmake/g++/clang++/npu-smi；`/usr/local/Ascend`、`/opt/Ascend`及`/dev/davinci0`、davinci_manager、devmm_svm、hisi_hdc均不存在，CANN相关环境变量未设置。Ubuntu未安装numpy/ml_dtypes。该结论是本次环境可见性，不断言整台电脑物理上绝无其他加速器。原始检查见`wsl-environment.json`。WSL首次沙箱枚举返回E_ACCESSDENIED，获工具审批后成功；没有安装软件或修改网络/安全配置。

使用Windows现有`E:/Anaconda/python.exe`独立运行P103存档工程的2项CPU检查和J001的3项CPU检查；原始结果及命令保存在`local-regression.json`与5份`.log`。这不执行编译后的Host/Kernel；本机现有条件无法完成设备编译与NPU测试。

## 在已有Linux/CANN/910B环境执行

先在本人工作空间分别解压P103原包与J001原包到两个新目录。分别执行工程已有构建命令并保存完整日志，使用各自新构建的包；不要复用另一个版本的build目录。下例CODE、DIAG、RUNS均替换为实际绝对路径；RUNS必须尚不存在。已安装numpy及ml_dtypes是生成器前提。

```bash
set -euo pipefail
CODE=/absolute/path/to/extracted/code
DIAG=/absolute/path/to/experiments/J002
RUNS=/absolute/path/to/new-p103-or-j001-run
# 先按本题要求source CANN set_env.sh；在CODE下执行 bash build.sh -j4
OPP="$CODE/build/packages/vendors/p_relu_grad_update_custom"
cmake -S "$DIAG" -B "$DIAG/build-p103-or-j001" -DCODE_ROOT="$CODE" -DOPP_ROOT="$OPP"
cmake --build "$DIAG/build-p103-or-j001" --parallel 4
python3 -B "$DIAG/diagnose.py" generate "$RUNS" --code "$CODE"
python3 -B "$DIAG/diagnose.py" run "$RUNS" --runner "$DIAG/build-p103-or-j001/device_probe" --opp "$OPP" --device 0
```

run为每次子进程指定本次自定义包路径，记录源码、runner和包文件hash及加载环境；保存逐case日志和两个输出。记录显式标注source_package_binding=UNVERIFIED，直到匹配的构建manifest证明源码→包关系；此记录不能替代平台构建指纹或证明Judge实际运行了同一个包。完整设备型号/CANN版本、实际Host tiling和生成同步代码仍需随设备日志保存。每次重跑用新的RUNS，避免旧输出混入。CMake每次清除两项库路径查找缓存；仍建议基线/候选独立构建目录。

先检查既有FP32控制组；若P103/J001都失败，查加载包/共同实现。若只J001的C=1失败，再缩小到最小长tile输入核查。三dtype有限诊断通过后，执行已有 `AI-precision-P103-validation-20261103.md` 的数值边界判别。原seed42样例也应在两份独立工程各跑一次；本诊断的同shape控制组并非原seed42数据。
