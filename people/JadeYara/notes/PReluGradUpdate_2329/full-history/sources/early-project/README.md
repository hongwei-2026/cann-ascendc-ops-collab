# JadeYara — PReluGradUpdate_2329

当前 workdir 候选：**J001，已在 CANN 9.0.0 / Ascend910_9362 原生编译，54/54 设备精度通过，并完成两轮本地性能对照。**

最新进度与测量见 [910C工作记录](experiments/J002/PROGRESS-910C-20261104.md)。J003已扩展至78/78精度；新增J004 S=1通道权重向量化候选完成96/96精度和两轮性能验证，详见 experiments/J004/RESULTS.md。两版独立保存在deliverables；原workdir仍为J001。以下为最初 J001/J002 记录，保留历史验证范围。

身份已通过连接的 GitHub profile 核实为 JadeYara。此目录为用户自己的实现；hongwei-2026、yuhongming-2026 目录仅作只读参考。

## 基线与单一改动

- 来源：协作仓固定提交 `22ec6d63da92ef2193dd66d1666f5d238cf3d08d` 的 hongwei P103，共41文件，下载后逐个核对 Git blob SHA。
- `baseline/P103-code.zip` 保留原始 `code/` 工程；`baseline/manifest.json` 记录每文件 Git blob/SHA256。
- `workdir/code/` 为 JadeYara 的 J001 候选。
- 生产改动仅 Host 两处分派：合法权重长度大于1才视作逐通道权重。C=1、weights=[1] 不再按 innerSize 截断连续 tile。
- Kernel、Compare/Select、数据搬运、dtype key 和舍入均保持 P103；没有合并每核权重缓存或 S=1 向量权重路线。
- 公开五点 shape 未知。只有命中 C=1 的场景可能受益，不能预测平台得分。

## 已完成验证

在 Windows Python 3 / numpy 2.4.6 / ml_dtypes 0.5.1 上：

```powershell
python -B workdir/code/tests/local_semantic_check.py
python -B workdir/code/tests/compute_schedule_check.py
python -B workdir/code/tests/shared_weight_check.py
```

三个 CPU 检查均 PASS；新检查816组、每模型1,519,116个元素，核对权重索引、每元素恰好覆盖一次、两输出与独立广播参考一致，并检查模拟UB预算。日志在 `experiments/J001/`。这是软件模型，未验证真实 BF16/FP16 设备行为。

## 下一闸：设备构建与同机 A/B

在题目指定的 Linux/CANN 9.1/910B 环境，分别将基线和候选放入独立目录；禁止复用 build/安装产物或覆盖其他创作者目录。依题包要求设置环境后执行：

```bash
python3 -B tests/local_semantic_check.py
python3 -B tests/compute_schedule_check.py
NLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include GTEST_SOURCE_DIR=/path/to/googletest bash build.sh -j4 -u
DEVICE_ID=0 bash build.sh -j4 -e
```

`-u` 已新增三dtype C=1连续分块及跨batch的四个实际Host tiling UT，本轮未运行。`-e` 仍是原FP32 [4,6,8] 样例，**不代替**三dtype泛化验证。

设备回归至少覆盖 `[128,1,1]`、`[32,1,17]`、`[9001,1,1]`、`[533,1,17]` 的三dtype，以及原 `[4,6,8]` 的 weights=[1]/[6]。需要参数化设备runner或正式Judge覆盖这些case，原example没有该能力。使用正式比较配置，保留原始输入/输出、源码和包hash。

性能用同机同shape/dtype、预热后交替重复P103/J001，分别记录kernel及端到端时间、原始样本与中位数；没有稳定收益则不采纳。`[160000,1,1]` 在48核模型里tile数160000→48只是调用数量估算，绝非加速比。

本轮没有 NPU 访问、未运行编译/UT/平台提交、未push。原E盘仓库HEAD仍为旧fc85a48，其已有未提交文件保留；本目录来自固定远端P103，不能用旧本地hongwei代码作为同机基线。

开源来源和适用范围见 `SOURCES.md`；会话范围见 `session-card.md`。


## J002 正确性调查补充

用户报告提交1109131为WA，上传物尚未绑定到本目录J001。参见[诊断记录与可运行工具](experiments/J002/README.md)。本轮没有改生产实现或原交付包；30组设备诊断已准备，只有CPU验证结果，设备验收仍待执行。


最新分派实验：J005 96/96、J006 45/45设备精度通过，四版同机性能及采用结论见 [J006结果](experiments/J006/RESULTS.md)。


输入预取实验J007：96/96设备精度通过，两轮性能和采用结论见 [J007结果](experiments/J007/RESULTS.md)。


缓冲复用实验J008：96/96精度通过，结果见 [J008报告](experiments/J008/RESULTS.md)。


权重缓存实验J009：102/102精度通过，三版本交替性能见 [J009报告](experiments/J009/RESULTS.md)。


分核实验J010：114/114精度通过，延迟和实际Block Num见 [J010报告](experiments/J010/RESULTS.md)。


规模阈值扫描选择J011目标下限4核，报告与曲线见 [J011](experiments/J011/RESULTS.md)。


J013 已整合 J011/J004/J008，273 个精度输入通过；性能和顺序敏感性见 [J013 结果](experiments/J013/RESULTS.md)。


J015 已验证对齐 S=1 权重复用收益，J013 保留综合基线；见 [J014/J015 对照及断点](experiments/J015/RESULTS.md)。


## J018 完成：驻留相位修复与完整归档（2026-11-04）

J018 单条件修复构建通过，6个失败边界的三dtype复核全部正确，435既有+27新增=462独立精度全部通过。J016同输入复现原4失败，新增相位再发现4失败，8项排除性能。J013/J015/J018各69输入、J016合格61输入，四轮共1072组/26800 kernel记录，输出全通过。Host规则计算28个驻留输入平均全部改善，27个每轮至少快0.15us；BF16 [1201,512,1] 16.384→10.720us，[127,4096,1] 10.880→9.517us。非驻留路径回退：BF16 [2001,512,1] 23.119→25.679us，小S=1也退化，故J018仅保留专项候选，不替换J013/J015。下一步候选是独立编译隔离驻留路径，本轮未实施。

J016/J017/J018完整云端证据已下载，SHA/CRC/全部二进制索引验证通过：J016-J018-910C-evidence.zip 354043635 bytes，SHA 2187aaced95a13649de82f014af9b5f0f672eb18cb92d1edccaf1eaf83b507ef。报告 people/JadeYara/PReluGradUpdate_2329/experiments/J018/RESULTS.md；归档 tasks/active/PReluGradUpdate_2329/explorations/logs/EXP_J016_J018_RESIDENT_910C_20261104。无在途NPU作业，无提交或push。


## J019/J020 独立编译实验完成

两版各498精度通过，1680+780+420=2880组输出全通过。完整对照见 people/JadeYara/PReluGradUpdate_2329/experiments/J020/RESULTS.md。

## 本轮结论与采用状态

**独立分派实现与精度通过，但尚未全面隔离小输入回退；J019/J020 不作为新的全量默认候选。** J013 保持综合基线，J015 与 J018 的原专项候选继续保留。本轮没有声称达到未知榜单的 4–5 μs 目标。

固定位置复核给出了更具体的边界：

- 大输入之前的 [3,5,1]，J020 相对 J013 的平均差值为 FP32 +0.028、FP16 +0.005、BF16 -0.025 μs，接近原水平。
- 驻留大输入之后，同一小输入 FP16 为 2.950→3.550 μs，BF16 为 2.970→3.592 μs；四轮差值分别约 +0.57～+0.63、+0.59～+0.64 μs。
- 非驻留大输入之后，FP16 小输入 2.948→3.347 μs，BF16 2.977→3.362 μs；四轮分别 +0.34～+0.43、+0.37～+0.40 μs。J019 也有相同现象。因此这不是只在长序列里出现的一次异常。
- 大输入的权重驻留收益保留：固定位置 BF16 [1201,512,1] 为 J015 17.637、J018 10.950、J019 10.802、J020 10.732 μs；FP16 同 shape 为 18.247、10.775、11.220、10.942 μs。新分派没有证明所有 dtype 都优于 J018。
- 非驻留 [2001,512,1] 的固定位置 J020 与 J015 相比：FP32 +0.056、FP16 -0.638、BF16 -0.185 μs；原 J018 大幅回退在此短批没有复现，但不能据此抹去早期测试差异。
- 600k 普通路径 J020 相对 J013 的固定位置平均差约 +0.075/+0.000/-0.045 μs（FP32/FP16/BF16），没有新算法收益。

编译期常量化不是普遍收益：J019 在 [40001,16,1] 的低精度路径有额外改善，但其他驻留路径的差值随 campaign 变化，J020 恢复原实现后也没有解决所有小输入后继开销。当前证据不能单独把问题归因于指令缓存、频率或某个同步事件。

固定位置复核控制了输入顺序和位置，但前序大输入在各版本中的实际执行时长不同（例如 J013 数十 μs、优化版十余 μs），没有把时钟、缓存或前序执行状态固定成相同值。因此低精度小输入的差异是该调用序列下的实测现象，不足以证明小输入核代码本身增加了同等开销。若要求跨序列都无回退，此验收条件尚未满足。

进一步读取已有 profiling：BF16 tiny_after_resident 的 J013→J020，Task Duration 为 2.970→3.592 μs，aiv_time 为 2.434→3.066 μs；FP16 对应为 Task 2.950→3.550 μs、aiv_time 2.413→2.998 μs。额外差异主要伴随设备 AIV 执行区间增长，下一步应从该区间查起，不能将其归于 SSH 或本机网络延迟。Scalar/Vector/MTE2 活跃比例仍相互重叠，不足以定位独占等待原因。原始聚合见 position-metrics.json。

后续应先定位“低精度小输入紧跟大输入”的稳定额外开销，结合入口反汇编、代码布局与流水线事件做一个有针对性的实验；不继续同时修改分派和数值路径。本轮没有进一步修改代码。


归档 tasks/active/PReluGradUpdate_2329/explorations/logs/EXP_J019_J020_DISPATCH_910C_20261104；增量归档与指定J016-J018完整归档共同校验通过，可恢复全部证据。无在途NPU作业、无提交或push。
