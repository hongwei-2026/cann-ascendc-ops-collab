# J037 平台与基线核验

本轮保持 J035 源码不变，未提交比赛。检查时间为 2026-11-05（设备日志为 UTC+08:00）。

## 在线题面

读取 https://cannjudge.cn/public/ictcompetition/prelugradupdate ：要求功能与 TBE 一致，覆盖支持的类型和格式，按逻辑值比较；浮点精度 float 1e-4、half 1e-3、int 0。页面未单独列出 BF16 阈值，未公布实际评测机型号/CANN 版本。当前可见题面没有旧本地 TASK.md 中的“95%”条款，因此不沿用该数字作为当前验收规则。公开样例不能代表隐藏输入。

读取本人历史提交 https://cannjudge.cn/public/ictcompetition/prelugradupdate/submission/6ae9f56a1dac5d235b45cfbf ：ID 1109131，2026/11/03 22:53:30，WrongAnswer；第一个公开点 WA，后四点 Skipped，0/5 公开点通过，另显示 45 个隐藏点。WA 点的 3.48 us 不能作有效性能基线。

下载工程 `PReluGradUpdate_submission_1109131.zip`，38 个文件，SHA256 `3b0c67f14e1921dee6f46482f22ee7b7da189a1d54be405c2cfd4f560f5b82ee`。与保存的 P103、J001 比较，排除 CRLF/LF 后仍各有 9 个同名文件不同，包含 host tiling、kernel 和类型注册。它不是本地保存的 J001；本轮未复现该历史 WA，不能据下载源码断言隐藏失败原因。完整逐文件哈希和 diff 已保存。

该提交 build.sh 默认 `COMPUTE_UNIT=ascend910b`，这是源码默认构建目标，不能据此判定平台实际硬件或构建时是否覆盖参数。本机实测目标继续为 ascend910_93 / Ascend910C，CANN 9.0.0。

浏览器等待下载事件曾长时间不返回，随后只读核对下载目录确认文件已实际保存，未重复下载。设备计时作业在此之后才启动，不把等待浏览器的时间算作 NPU 测试时间。

## 直接 TBE 路径

使用已安装 CANN 的 `aclopCompileAndExecute("PReluGradUpdate", ..., ACL_ENGINE_SYS, ACL_COMPILE_SYS, ...)`，测试程序只链接 acl_op_compiler、ascendcl、nnopbase。子进程移除 ASCEND_CUSTOM_OPP_PATH，未链接 cust_opapi；系统文件未修改。

安装源码 `opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/prelu_grad_update.py`，SHA256 `c97cc823da4208f11f59a491231149c0b81105f6740bed9a707c28f555f9f9cf`；ascend910_93 legacy 注册中存在 PReluGradUpdate。源代码、注册 JSON、API 头和 runner 保留快照。

小试 fp32 [4,6,8]、权重 [6] 通过；其后六个大输入 TBE 输出均与同一 golden 逻辑值精确一致。测试为有限随机值及零；没有将这些结果扩张为所有格式或 Inf/NaN 通过。系统 TBE 乘零路径与直接 Select 的 IEEE 特殊值传播可能不同，仍应在平台明确输入域后核验。

计时统一 ordinary msprof，PipeUtilization，Task Duration(us)；每个输入 250 次，取第 101–250 次的中位数。四轮 ABBA/BAAB，TBE 对照末两轮反转输入顺序。报告以每轮两个中位数的均值比较，再汇总四轮；不是把所有样本混为一个中位数。比较的是设备 kernel 任务时间，不包含两种 host API 调用路径的端到端开销，也不等于平台评分口径。

TBE 与 J035 同用 ND 输入、逐通道一维权重，六个输入：三 dtype [127,8192,1]、fp32 [91,10368,1]、fp32 [147,6656,1]、fp16 [101,8256,1]。复用 J035 保存的精确输入和逐文件 SHA，不重新抽样。

三个争议点另做 J034/J035 单 shape 进程、四轮 ABBA/BAAB；两版本同一输入、同样 250 次及采样窗口。它可以检验同一输入在隔离顺序下是否仍翻转，不能证明所有运行环境无波动。


精确输入来源：

| dtype | shape | 数据集/seed |
|---|---|---|
| fp32 | [127, 8192, 1] | seed_202611050024 |
| fp16 | [127, 8192, 1] | seed_202611050024 |
| bf16 | [127, 8192, 1] | seed_202611050024 |
| fp32 | [147, 6656, 1] | seed_202611050026 |
| fp16 | [101, 8256, 1] | seed_340101 |
| fp32 | [91, 10368, 1] | seed_350091 |

本机构建证据：remote_build_j035.sh 显式使用 -DASCEND_COMPUTE_UNIT=ascend910_93，按 all/binary/package/install 四目标生成个人目录下的软件包。没有为平台未披露的目标新建猜测性构建。
