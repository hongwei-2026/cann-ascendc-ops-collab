# Dev log — ConfusionTransposeD_2323

## 2026-11-12（P009修订版云测）

- `0878104`：基线与候选样例均0/120、各39/39 bitwise PASS；候选边界46 PASS、6 FAIL_OR_REJECTED。
- 39项tiling输入、属性、输出存储shape检查通过；39项infer未观测，不能称infer验证通过。控制台摘要已归档，原tar仍在云端。
- 下一闸平台日志可见性，真实平台shape和隐藏点RE原因仍未知；不改性能Kernel。

## 2026-11-12（P009探针修订）

- 补充回传确认两例只见tiling事件；CANN头文件声明tiling可读输出StorageShape。归档用户回传摘要。
- 增加tiling输出存储shape记录；检查器验证fixture匹配并单列未观测infer，不能把输出描述当成infer通过。Kernel与算子判断不变；修订版待云端编译和NPU验证。

## 2026-11-12（P009 云测首轮回传）

- 归档CodeArts控制台证据段；候选 `ed583cb` 与P004均样例PASS、39/39 bitwise PASS。边界候选46 PASS、6 FAIL_OR_REJECTED。
- 检查器因39项未观察到成功infer事件而exit=1；每项观察到匹配tiling成功事件。探针验收未完成，不能由此推断infer回调未执行。
- 原tar仍在云端未下载；下一步复用逐case日志及CANN头文件检查输出shape可得性，平台case和隐藏点RE仍未知。

## 2026-11-12（P009 Host探针）

- 在P004 Host infer/tiling入口及退出写`CTD_PROBE` JSON，记录输入、属性、dtype枚举、成功/拒绝阶段及可得路径/UB信息；Kernel和算子判断未改。记录是回调级而非保证一条对应一个平台case。
- 本地语义检查、脚本语法、模拟Host JSON成功/失败事件检查通过；未在本机编译CANN或运行NPU。新增CodeArts隔离P004/候选样例、39项bitwise、边界诊断及探针解析流程。
- 平台stderr可见性及隐藏点是否进入回调未知；等待C009云测，再判断提交探针包。没有平台shape或系统TBE对应profile，不作验收结论。

## 2026-11-12（平台再提 + 探针交接）

- P004 再提：Runtime Error；1/4/5≈717–725μs，2/3≈2.03–2.05ms，点6隐藏。
- 写 [HANDOFF-PROBE-SHAPES.md](HANDOFF-PROBE-SHAPES.md) 给其他 AI：优先探针探形；登记 P009。
- 微优化仍按 DEC007 冻结。

## 2026-11-12（P008 msprof + DEC007）

- msprof：`te_confusiontransposed_*` / OP `ConfusionTransposeD`；op_summary last10 **mean=4.804μs** @ `(2,1024,128)`。
- 写 DEC007：停 Host/P004 微优化；等 Owner 是否立项镜像 TBE 模板。
- 归档 `C008-msprof-system-paste.log`；推 main。

## 2026-11-12（P008 Host 尺度）

- 四档 Host ~0.09–0.10ms 贴死 → Host 基线作废。归档 `C008-torch-npu-scale-host-paste.log`。
- 下一闸：单 case msprof device task；失败写 DEC。

## 2026-11-12（P008 系统可调用）

- `npu_confusion_transpose` + CUSTOM unset：sample 出 `(2,4,3,5)`，Host 0.0953ms/call。
- 归档 `C008-torch-npu-system-call-paste.log`。非 msprof；下一闸同输入更大 shape / 可选 P004 A/B。

## 2026-11-12（P008 系统符号）

- 系统有 `aclnn_confusion_transpose.h` + `libopapi` 导出无 D API；无系统 `*_d.h`。
- `torch` / `torch_npu` 均可用。归档 `C008-system-symbol-paste.log`。
- 下一闸：torch_npu 同输入调用（勿跑 examples/run.sh）。

## 2026-11-12（P008 样例库存）

- CodeArts：`git pull` 易挂在 Receiving；可跳过。`CUSTOM` unset + `find` 已确认有 `examples/test_aclnn_confusion_transpose_d.cpp`。
- **禁止**直接 `examples/run.sh`：脚本会重新挂自定义 OPP，且 API 是带 `D` 的自定义入口。
- 下一闸：nm/strings 查系统 `aclnnConfusionTranspose`（无 D）+ torch_npu 有无。

## 2026-11-12（P008 API 签名回传）

- Owner 粘贴 `aclnn_confusion_transpose.h`：`perm/shape/transpose_first` 齐全；TBE apt 注册 ConfusionTransposeD；device 分派六类 Transpose*（arch35）。
- 归档 `explorations/logs/EXP_C008/C008-api-signature-paste.log`。判定：**签名闸过，对照闸未过**；活动 Kernel 仍 P004；**不可**当新优化粘平台。
- 下一闸：不加载 custom opp，对 P004 样例调系统 `aclnnConfusionTranspose` + msprof（≤30min）；失败写 DEC。

## 2026-11-11（P008 CodeArts回传）

- CodeArts盘点exit=0，机器为910B3、CANN9.1.0，`ASCEND_CUSTOM_OPP_PATH`未设置。确认CANN OPP含legacy ConfusionTransposeD TBE Python、910B legacy注册配置，以及dynamic TBE/AscendC实现源码。
- 扫描命中的Transpose/Gather为ACLNN/ES接口，不能据此声称存在AscendC Kernel级Transpose/Gather原语；TBE独立调用和同输入device task profile仍未证明。平台点shape也未知。
- 原tar位于CodeArts且未下载；归档用户粘贴marker段到`explorations/logs/EXP_C008/C008-console-evidence.log`，SHA256 `B12C65694536E0D305DD920A6EE42ED0DC10F41CE861D97C5276D736649A9C03`。P008在30分钟内继续核对TBE调用入口，不改Kernel。

## 2026-11-11（P008 TBE源码入口回传）

- 源码输出确认TBE的AscendC入口使用`get_code_channel`注册编译；设备kernel按tiling key分派TensorMove、SmallShape、CutOneAxis、CutTwoAxis、BigDim、NLast六类Transpose模板。
- 找到`aclnnConfusionTranspose`头文件，但当前输出没有完整签名；未发现`aclnnConfusionTransposeD`。尚不能判断公共API是否具有`perm`、`shape`、`transpose_first`全套语义，也没有同输入TBE profile。
- 用户粘贴源码输出归档至`explorations/logs/EXP_C008/C008-source-probe-paste.log`，SHA256 `E3F8D76CA63F20C5B74B49314D9CBD348AFB98776CFF4FB722197236B9451F32`。下一步只核对API签名和TBE模板头依赖；Kernel继续冻结。

## 2026-11-11（P008前置盘点）

- Owner要求继续；P006/P007均止损后，没有把排行榜best time当TBE基线。P008仅盘点CodeArts的CANN 9.1 TBE实现、Transpose/Gather API和可复现输入，预算60分钟；Kernel暂不改。
- 新增`tests/run_c008_preflight.sh`与`C008-CLOUD-PREFLIGHT.md`、`EXP_C008-tbe-api-preflight.md`。等待Owner在CodeArts运行并回传`C008_EVIDENCE` marker或完整tar。
- 若无法单独调用TBE并测得同机device task时间，按P008止损写DEC，不开Kernel优化路线。

## 2026-11-11（C007止损）

- 归档C007粘贴证据段，SHA256 `816dfb66e398b839b5a9f2fc573ad3668a045b64abad774d70bccd1c97ea2911`；CodeArts原tar未下载，证据范围如实标注。
- 910B3/CANN9.1.0，P004和P007均build完成、39/39 bitwise PASS；四个msprof案例均25次有效Kernel及精确输出。目标128行6.72→6.56μs，仅1.024×；小样4.08→4.36μs，回退6.86%；rank2 8.30→7.82μs。未达到1.3×目标且触发2%回退上限。
- 按DEC006关闭P007，不开P008；Kernel、Host tiling、tiling data恢复到P004 `4761239`。平台Runtime Error仍未定位。

## 2026-11-11（C006结案 / P007有界跟进）

- 归档用户粘贴的C006 marker证据段，SHA256 `9cec05afe07c563f37462b6ccb05766d11d7e4841ea34e8342e6e1d60461e3e0`；原CodeArts tar无法下载，本次不称完整原始日志已回收。
- 910B3/CANN9.1.0：P004与P006均39/39 bitwise PASS；7项有效msprof中P006的5个步长目标全部慢于基线，rank2与小样近乎持平，128行案例也从7.170降至7.520μs。P006假设失败，详见EXP_C006与DEC005。
- 活动Kernel回到P004基线`4761239`；新增且仅新增P007：规律源步长的连续输出行合并MTE2块DMA。期限90分钟本地实现/语义、云回传后30分钟分析；128行目标≥1.3×且rank2/小样回退≤2%，否则DEC止损。
- P007实现限定在32B整行且源行起点形成正向等差组；起点进位时重分组，其余保持P004行搬运。local semantic、P004计划1658项、P007计划266项通过；128行CPU模型输入搬运次数128→32；Bash脚本语法检查通过。仍未有CANN编译、NPU回归或P007性能结论。
- 默认Git CA bundle的pull因issuer证书失败；改用Windows系统信任证书库重试后成功，`origin/main`无新提交，基线仍为`26a985d`。

## 2026-11-11（P006步长DMA候选）

- Owner授权继续优化；同步远端无本题冲突，保留他题归档；先登记P006/SESSION假设、90分钟实现/30分钟云回传止损，再修改hongwei本题Host/Kernel。
- span=1按末个非单元输出轴成批步长DMA，32B UB槽→MTE3密集写回；Host按真实槽容量和32B核边界分配。连续后缀路径保留，不改dtype key，不混超UB/跨rank TF/-1功能修复。
- CPU原语义、1658旧计划、1002新地址/字节oracle通过；39fixture仅生成，无dtype跳过，未执行CANN/NPU。
- 准备隔离P004/候选两套样例-e+39项bitwise+7项msprof；关闭DEBUG洪水，完整日志进tar、终端直接给可复制报告，另将tar复制到CodeArts项目文件树便于下载。
- 下一闸Owner执行C006，按同机证据结案/DEC；当前没有P006性能数字或平台通过结论。

## 2026-11-10（C005本地原因结案）

- 第三份粘贴61016字节原文及解析归档，SHA256 541617ff…c310646；六份日志报告哈希与上次一致。
- candidate明确Host拒绝：UB整输入上界3项、perm/shape rank不符1项、shape总数不符2项；六项均tiling失败→ACL561002，未launch。两版源码均有对应检查，baseline正文未取得，不冒充双版本根因日志齐全。
- DEC004关闭P005追加诊断，不再要求Owner重复复制日志；平台RE原因仍未知，无新性能收益或验收PASS。
- 本次只更新本题台账/证据，Host/Kernel/测试代码未改；功能限制修复与性能路线须分别新登记P行、最小实现后云测。

## 2026-11-10（C005错误阶段补录）

- 第二份粘贴52713字节原文与解析归档；12条失败记录均exit=2，停在GetWorkspaceSize调用范围，未开始launch。平台RE原因仍未知。
- 发现六份candidate日志各约43MB，旧工具首尾摘录漏掉中间错误；仅维护只读导出工具≤15分钟，新增全文错误筛选与行号/上下文/省略计数，不重测设备。
- 冷进程host时间来自DEBUG全开诊断，不能作为性能数据；继续冻结Host/Kernel，只补录已有证据。
- 只读工具本地验证：扫描超过43MB的模拟日志，成功提取中间错误/ACL返回码；匹配超量时显式计数，两版CLI导出通过。记录`Q9AC/error-filter-local-check.json`，非云端/NPU PASS。

## 2026-11-10（C005 Q9AC控制台回传）

- Owner无法下载tar，归档7093字节控制台原文及可复核解析JSON；SHA256 bd877ba7…b126835a，完整tar/逐case错误尚缺，不记完整日志归档。
- 原版/P004均46/52逐字节通过，同6项未通过；52项结果相同，无已观察到的新增P004失败。大输入3项、TF跨rank及两种-1属性需读失败阶段，平台RE仍未知。
- 修正空tensor静态推断：端到端实测两版通过，不能推断Kernel被调用。没有新的性能数据。
- DEC003有限证据结案；增加只读文本导出工具，供Owner复制已有Q9AC失败详情/host阶段，不重编、不重跑、不改Host/Kernel。
- 本地原语义检查通过；文本导出语法/摘录标识/全文及缺失用例检查通过（仅CPU工具验证），原文哈希核对通过。无新的NPU执行或性能数据。

## 2026-11-10（平台重提RE / P005诊断）

- 归档平台截图：点1/2约717/728us，约2.4×改善；3～5仍1.76～1.84ms；输出均正确，但总状态Runtime Error。无失败点/shape/错误详情，不认定点6。
- DEC002停止以本机22/22代表平台通过；先登记P005，仅测试/诊断，不改Host/Kernel，不做超UB分块。
- 准备52项边界探测，复用C004两套已编译包，源码核验，按API阶段记完整错误及主机时间；安装CANN原语头文件只读快照供后续选路。
- 本地生成/语法/原语义检查通过；诊断待Owner云端执行并回传完整tar，不宣称错误已修复。

## 2026-11-10（P004 云日志回收/结案）

- 回收 EXP_C004_20261110_142548_L3wI.tar.gz（SHA256 14c8db6d…447365d5），完整原包和285份文字证据进本题日志目录；源代码与545f8d2/4761239逐个核对一致。
- 910B3/CANN9.1.0 exit=0；原版/候选均样例PASS、22/22 bitwise PASS，含bf16与int64，无跳过。
- 原始CSV重算：样例8.50→4.06us；half24.43→10.84us；长行41.07→6.77us；rank2 8.26→8.32us。采纳2.09～6.07×有限收益，≥10×假设未达。
- DEC001 冻结本轮Kernel，先平台重提；平台1.7ms与本轮原版8～41us的对应关系未证实，不宣称400×。只修日志解析器的Block Num识别，未改Kernel。

## 2026-11-10（P004 候选，待云测）

- 同步 main=545f8d2；只在隔离副本的 hongwei 本题座位实现；补事实 SUMMARY/会话卡片，先登记 P004 连续行 DMA 假设与时限。
- 连续后缀走 MTE2→MTE3 整行批量 DMA，短行 UB 补齐后密集写出；span=1 保留原实现；超 UB tensor 边界不扩大。
- CPU local semantic + 1658 项 copy-plan PASS；22 项 fixture 生成、脚本语法检查通过。**无 CANN/NPU PASS 或性能收益结论**。
- 云测由 Owner 手动执行；一键脚本对比 545f8d2/候选，失败也打包；见 C004-CLOUD-RUN.md。下一步回收云日志、按时限结案/DEC，禁止无证据转路线。

## 2026-11-10（性能交接）

- 平台：点 1～5 功能 Pass、错误 0%；用时 ~1.7ms vs 最优 ~3.5μs（~500×）→ STATUS **性能中**
- 记 EXP_C003 基线；path-ledger **P003**；写 `HANDOFF-PERF-PROMPT.md` 供其他 AI 有界优化
- 根因假设：Kernel 标量 `GetValue`/`SetValue` gather；下一 AI 按提示词开 A/B 路，勿无台账死磕

## 2026-11-10

- 授权：拷贝座位到 `people/hongwei-2026/ConfusionTransposeD_2323/`，先做扩泛化/云测（不改 yuhongming workdir）
- path-ledger **P002** 开工；加泛化/云测脚本 + `SYNC_CODEARTS.md`；`207ea78` push
- **EXP_C002 云测 PASS**：exit 0；样例 0/120；回归 13/13；日志归档 `explorations/logs/EXP_C002*`
- 原「下一闸超 UB」后置，优先性能

## 2026-11-04

- 从 `ConfusionTransposeD_problem_2323_template.zip` 建座位与台账
- 实现 infershape / tiling / kernel（输出下标 → 输入下标）
- 本地 910B4 `bash build.sh -e`：**PASSED**（y mismatched=0/120）
