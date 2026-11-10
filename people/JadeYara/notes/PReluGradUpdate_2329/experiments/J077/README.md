# J077：真实 TBE 契约、UT 与目标包复现修复

日期：2026-11-08。基线：官方提交 1221802，原始 48 文件保存在 baseline/code。op_host 与 op_kernel 均未修改。本轮只修数值参考、UT 和构建入口。

## 已确认的结果

| 检查 | 结果 | 证据 |
|---|---|---|
| Python 数值链单测 | 7/7 通过 | CPU 单测日志、sample93/UT 日志 |
| SDK Host UT | 9/9 通过 | logs/ut-attempt2.log |
| SDK Kernel CPU 模拟器 UT | 1/1 通过，dx/update 均 0/192 不匹配 | logs/ut-attempt2.log |
| 显式 ascend910_93 构建与设备样例 | 通过，dx/update 均 0/192 不匹配 | logs/sample93-attempt1.log |
| Python 模型对已归档真实 TBE 输出 | 67 个已接受输入，逻辑差异与阈值差异均 0 | model-vs-actualTBE.json；输入、meta、金标准哈希均核验 |
| TBE 拒绝的描述 | 2 个仍列为拒绝，不计通过 | probe_scalar_grads、probe_rank1_features_scalar_weight |

CPU 模型对照使用 J076 归档的真实设备 TBE 金标准，并非把 CPU 计算当成新的设备执行。Kernel UT 使用 Ascend910B1 CPU 模拟器；设备样例运行于 Ascend910_9362。结果范围分别报告。

## 默认 910b 失败的根因

当前设备向 runtime 报告 Ascend910_9362。默认包只提供 ascend910b 的内核目录与注册 JSON。运行时加载自定义 libcust_opmaster_rt2.0.so，却选择了 SDK built-in 的内核：

`opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy/prelu_grad_update/PReluGradUpdate_2dc012f1f3b7976934d280e17d5c376e_high_performance.o`

自定义 Host 生成 80 字节 tiling、key=0、24 blocks，与选中的内置 TBE 内核混用；执行报 ACL 507035，设备报 scalar 内部缓冲访问越界。日志直接证明 Host/Kernel 来源混用；这里未声称解码过内置 TBE 的每个 tiling 字段。

控制对照使用同一个样例可执行文件、相同生成输入、相同设备，只更换独立包：

| 包 | 运行/校验退出码 | 实际内核 |
|---|---|---|
| A：历史原始 910b 包 | 2 / 1 | SDK built-in TBE |
| B：原生 910_93 构建 | 0 / 0 | 自定义 PReluGradUpdate_40c7… |
| C：A 的副本，仅改目标目录、文件名及 JSON 标签 | 0 / 0 | 自定义 PReluGradUpdate_40c7… |

C 的所有 .o/.so 哈希与 A 相同。历史 910b 与原生 910_93 包中 9 个内核 .o 也逐字节相同。由此定位为目标包查找与回退问题，不能归因于这 9 个内核的机器码不兼容。C 只是因果诊断，不作为通用跨型号包改名方案。

修复后的 build.sh 支持 `--soc ascend910_93`，并把目标纳入构建指纹；保留官方默认 ascend910b，使用者应选择实际设备的目标。当前实例的复现命令：

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}
export GTEST_SOURCE_DIR=/usr/src/googletest
export NLOHMANN_JSON_INCLUDE_DIR=/home/developer/opensource/json/include
bash build.sh --soc ascend910_93 -j4 -u
DEVICE_ID=0 bash build.sh --soc ascend910_93 -j4 -e
```

## 数值约束与 UT 修订

数值链保持 `dx=(pos+neg*w)*g`，先按输出 dtype 量化 dx，再计算 `update=(neg*x)*g+float32(dx)*0` 并量化输出。不能以 where 的分支跳过 0×Inf/NaN。FP16 常规 kernel 原生 half 路径与 BF16/General 的 float32 中间路径分别保留；本轮不改设备算术。

Host UT 用实际当前 metadata、key、core 数与广播规则替换旧模板期望，覆盖三种 dtype 常规样例、标量权重、General 广播、空输出与非法描述。Kernel UT 零初始化实际 tiling，明确 C/S/tile/weightsPerChannel/rows/split 字段，并使用正确的 float 模板与 key。模拟器 tiling 复制按实际类型大小进行。

首次 UT 构建失败已保留：配置 GoogleTest 后才把构建类型设为 Debug，生成 libgtest.a 却链接 libgtestd.a。将构建类型设置前移后，第二次通过。未删除失败日志或跳过失败断言。

数值模型核验脚本的前两次路径读取失败也保留：历史输入哈希键有短名和相对路径两种，NC1HWC0 的 meta 首组为物理形状、次组为原始形状。第三次按实际格式读取后完成 67 项校验，数值容差未修改。

## 决定

接受 J077 作为复现/测试修复。无性能结论，无官方提交。官方基线保持原样，后续优化单独记为 J078。


> 2026-11-10 同步注：本页为原实验记录。仅公开选定文本证据，不表示所有相对路径产物均已上传；代码状态和下一步以[当前进度](../../progress-20261110.md)为准。
