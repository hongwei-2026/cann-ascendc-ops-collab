# J056：仅 resident 多 tile 启用 C=64 广播

2026-11-06。设备验证完成，保留为独立试测候选；J035不变。未进行官方提交，未提交或推送公开仓库。

## 改动与范围

- 唯一计算改动沿用J055：读取一个旋转的64元素权重周期，Mul的src1RepStride=0；低精度权重只转换一次。最终候选选C，权重队列64元素。
- 仅既有resident模板（schMode 12–15）且C=64、S=1、逐通道权重、rowsPerTile>0、blockFactor>tileLen时启用。Host分核、tile与其他路径保持J035。
- 独立候选目录：candidate/code/。相对冻结J035的46个原文件，仅op_kernel/p_relu_grad_update.cpp改变，新增op_kernel/p_relu_grad_update_period64.h；其余45个原文件逐字节相同。未改原workdir。
- 当前910C：物理UB196608B，保留8192B后按原Host预算；FP32 tile=5184，FP16/BF16 tile=4288。阈值按Host公式推导，不按官方测试点特化。Mul repeat最大81/67，未触及uint8上限。

## 构建与正确性

- 同一Ascend910_9362、CANN9.3.0、compiler/OPP9.3.0-beta.1、driver25.5.5。私有包构建，无系统安装。A/B/C与独立正式算子包all/binary/package/runner均exit0。
- 新边界套件36个shape/dtype，108窗口、720次调用，两个输出全部通过。覆盖阈值前/等于/后、旋转通道及8或16元素短尾、两tile整尾、四tile短尾、小C64、C128 resident、C1、共享权重和S>1。
- 独立正式算子候选再运行780个已有精度用例+36个新边界用例：816/816通过，每次检查dx/update。保持原严格比较，仅允许正负零等价。
- A的45个函数体保持冻结J035一致；独立候选45/45个函数体与已测C逐字节一致。B/C部分回退函数长度相同但字节变化，不能声称全部回退二进制相同。
- 两次测试启动失败均发生在设备测试前：CRLF脚本、同名小输入目录。已修复，initial-*和plan-failed-*保留全文。

## 性能

同进程、固定五个张量及流，六种平衡顺序ABCA/CAAB/BACA/ACBA/BAAC/CABA；每窗口350次，舍弃200次，保留150次Task Duration中位数。864窗口、302400条内核记录、216组对照。六轮是进程内顺序，不是六次独立重启。

| 目标 [4097,64,1] | A J035 μs | C J056 μs | 改善 |
|---|---:|---:|---:|
| FP32 | 8.380 | 6.780 | 19.09% |
| FP16 | 8.380 | 6.806 | 18.79% |
| BF16 | 8.460 | 6.860 | 18.91% |

12个实际启用输入均六轮改善：FP32约0.850–1.600μs，FP16约1.540–1.749μs，BF16约1.520–1.760μs。C64小输入与阈值下方恢复J035分派，未重现J055的小FP32系统性退化。

**严格控制门槛不是全通过。** A/A最大0.0295μs，216组均稳定；未改路径四路极差门槛（≤0.1μs且≤2%）212/216组通过。4个例外位于C1的FP32/BF16第2、3轮，绝对极差0.050–0.060μs，在约2.3–2.5μs输入上略超2%。最终C相对A均值的全部回退对照仍小于2%（最大绝对1.913%），但不改变或追认原门槛。保留此结果，不为了刷过门槛重复测量。

因此交付独立试测候选，继续保留J035，不宣称已满足全控制组晋升条件，也不宣称官方提升。官方输入未知，由用户自行决定是否试交。

## 证据与复现

- code/、control/、plan.py、build.sh、tests.sh为A/B/C实验入口。候选为candidate/code/，candidate/build.sh和candidate/validate.sh为私有构建及验证入口。脚本中的远端根是本次环境，需要换环境时显式修改。
- evidence/下保留全部构建/失败/成功文本日志、precision/validated.json、perf/summary.json、overview.json、function-bodies.json及candidate/validated.json、candidate/function-correspondence.json。
- source-comparison.json记录候选与冻结J035的逐文件校验。J056-candidate-code.zip只含独立code/和简短说明，供用户本地取用。
- 已取回并校验文本证据：359355字节，153文件，SHA256 2b05bd7d7d8540119e12dbbd322a9be5905622c32c988bf1940a116287e20098。
- 完整原始输入及profile归档保留在远端 /home/developer/2329-JadeYara-20261104-1812/J056-evidence.zip，341335200字节，SHA256 836c2b74476469a57bf9942ce11d8af2a88798a6725cd2dc67c5d5b23127ec77。该341MB完整包未下载；所有成功/失败文本.log已下载。
