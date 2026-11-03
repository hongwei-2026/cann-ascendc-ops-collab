# MultilabelMarginLoss（problem 2327）算子设计文档

> 状态：设计草案，待任务方评审及 910B 实测确认。依据本地 `MultilabelMarginLoss_problem_2327_template.zip` 中的 `code/samples/reference.py`、`case.json`、OpDef 和 Ascend C 模板编写。题包未附正式任务说明、完整测试矩阵或性能日志；本文明确区分题包直接给出的契约与待确认假设。

## 一、需求背景

### 1.1 需求来源与目标

本地题包提供 `MultilabelMarginLoss` 自定义算子工程，目标芯片配置为 Ascend 910B。`code/README.md` 给出的开发环境为 CANN 9.0.0 或 9.1.0，并提供 ACLNN 样例、Python 参考实现和首个 NPU 样例。当前 Host 的 InferShape/Tiling 及 Kernel 主体仍为 TODO，工程处于实现前阶段。

设计目标是在保持题包参考实现可观察语义的前提下，以 Ascend C 实现双输出：逐样本或归约后的 margin loss，以及每个类别在 target 有效前缀中的出现次数。初版优先保证 shape、标签截断、重复标签、归约和 dtype 正确，再依据目标机逐用例性能数据优化。

### 1.2 对标实现与语义来源

本地 `reference.py` 注明其语义用于对齐 910B 内置 TBE 实现，并记录了两项与常见 PyTorch 说明容易混淆的行为：

1. `is_target` 是出现次数，不是 0/1 标志；一个样本内同一类别出现多次时，输出计数也大于 1。
2. 每行只处理 target 中第一个负数之前的前缀；该负数及其后的值不参与计数或 loss。

该参考代码还记录了测试观察：例如 `[3,2,0,3,0,0]` 产生类别计数 `[3,0,1,2,0,0]`；`[3,3,3,3,-1,...]` 中类别 3 计数为 4。由于题包没有附这些实验的原始 NPU 日志，实现前仍需在目标 CANN/910B 环境重放关键语义用例，确认当前评测基准仍与参考代码一致。

### 1.3 题包现状

| 模块 | 当前状态 | 设计处理 |
| --- | --- | --- |
| `op_host/*_def.cpp` | 已声明 x、target、y、is_target 和可选 reduction；芯片配置为 `ascend910b` | 保留契约，校验 reduction 和输入 shape |
| `op_host/*_infershape.cpp` | 仅按输入下标复制 shape，y 与 is_target 推导均不正确 | 按 reduction 推导 y；按 x 推导 is_target；补输入校验 |
| `op_host/*_tiling.cpp` | totalNum、ubFactor 为占位值，blockDim 固定为 1；dtype 只有两路选择 | 重写切分、资源预算、归约策略和 dtype tiling key |
| `op_kernel/*.{h,cpp}` | Kernel 主体为 TODO；当前入口只实例化 half 和 float | 实现 target 计数、pairwise hinge、样本/批次归约；补独立 BF16 实例 |
| `samples/case.json` | 仅覆盖 x/target `[4,6]`、float32、reduction=`mean` | 保留为首个端到端样例，补充独立边界和隐藏用例矩阵 |
| UT | tiling 期望值仍来自初始占位实现；kernel fixture 只构造首个小样例 | 修改 tiling 后同步 fixture，并补语义覆盖 |

## 二、需求分析

### 2.1 数学语义

令 `x` 的形状为 `[N,C]`，target 的形状为 `[N,K]`。对每个样本 `i`，定义有效前缀长度 `L_i`：从 `target[i,0]` 开始扫描，在第一个 `< 0` 的位置停止；若没有负数，`L_i=K`。负值位置及其后的元素全部忽略。对 `0 <= t < C` 的前缀标签计数：

```text
cnt[i,t] = |{ k : 0 <= k < L_i 且 target[i,k] = t }|
is_target[i,t] = cnt[i,t]                 // int32，保留重复次数
```

前缀内满足 `target[i,k] >= C` 的标签不映射到类别，计数时忽略，扫描继续；只有负值会终止前缀。这一规则来自 `reference.py` 中的 `(seg >= 0) & (seg < C)` 过滤。

对每个样本，非目标类别 `j` 满足 `cnt[i,j] = 0`，目标类别 `l` 满足 `cnt[i,l] > 0`。逐样本损失为：

```text
loss[i] = (1 / C)
          * sum(j: cnt[i,j] = 0)
            sum(l: cnt[i,l] > 0)
              cnt[i,l] * max(0, 1 + x[i,j] - x[i,l])
```

这等价于对有效前缀中每个合法目标标签位置分别计算一次 hinge loss；重复目标标签会重复贡献。分母是类别数 `C`，不是有效标签数或非目标类别数。若该样本没有合法目标类别，或没有非目标类别，loss 为 0。

`reduction` 定义如下：

| reduction | y 的值 | y shape |
| --- | --- | --- |
| `none` | 每个样本的 `loss[i]` | `[N]` |
| `mean` | `sum(loss[i]) / N` | `[1]` |
| `sum` | `sum(loss[i])` | `[1]` |

题包默认值为 `mean`。`case.json` 将单元素归约结果声明为 `[1]`，不能返回 rank-0 标量。

### 2.2 接口契约

| 参数 | 方向 | dtype | format | shape | 约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入 | float16 / float32 / bfloat16 | ND | `[N,C]` | rank=2，`C>0` |
| `target` | 输入 | int32 | ND | `[N,K]` | rank=2，batch 维等于 x；`K` 可按参考语义独立于 C |
| `reduction` | 属性 | string | - | - | `none`、`mean`、`sum`；缺省 `mean` |
| `y` | 输出 | 与 x 相同 | ND | `[N]` 或 `[1]` | dtype 随 x；按 reduction 推导 |
| `is_target` | 输出 | int32 | ND | `[N,C]` | 每类目标出现次数，逐元素精确 |

OpDef 通过 dtype 列表位置绑定 x/y 的三种浮点类型；target 与 is_target 固定为 int32。混合 x/y dtype 不在接口内。题包首个测试点为 `x=float32[4,6]`、`target=int32[4,6]`、`reduction=mean`，输出 `y=float32[1]` 和 `is_target=int32[4,6]`。

### 2.3 需求拆解

1. 校验输入 rank、batch 维、类别维和 reduction 属性；拒绝不支持的输入并返回明确错误。
2. 按行扫描 target 前缀；第一个负值截断，合法标签累计 int32 次数，越界正标签忽略。
3. 逐行计算非目标类别与目标类别的 hinge loss，重复目标按次数贡献。
4. 支持 `none`、`mean`、`sum` 三种输出 shape 和语义。
5. 支持 float16、float32、bfloat16 输入/损失输出，以及 int32 计数输出。
6. 控制 UB 占用，不构造 `[C,C]` 的完整 pairwise 临时矩阵；尾块不得越界。
7. 为每种 dtype、reduction、target 边界和大类别数设计可执行验证用例；性能结论必须来自目标硬件实测。

## 三、详细设计

### 3.1 使能方式

使用题包生成的 ACLNN 接口 `aclnnMultilabelMarginLossGetWorkspaceSize` / `aclnnMultilabelMarginLoss`。输入、输出、reduction 属性顺序按生成头文件和 `examples/test_aclnn_multilabel_margin_loss.cpp` 为准。首个样例目前只调用 `mean`；实现后需通过样例脚本核对 y 的 `[1]` shape、输出 dtype 和 is_target 的精确计数。

### 3.2 Host 侧：OpDef 与 InferShape

#### 3.2.1 OpDef

保留题包现有 dtype 与 ND format 声明：

- x/y：float16、float32、bfloat16，三者按列表索引一一绑定。
- target/is_target：int32。
- reduction：可选 string，缺省值 `mean`。
- AICore 配置：`ascend910b`。

当前 kernel 入口把 `DT_FLOAT16` 和 `DT_BF16` 共用 `half` 模板实例，这会把 BF16 位型按 FP16 解释。实现时必须设置三个不同 tiling key，并分别实例化 `half`、`float`、`bfloat16_t`；不能让 BF16 走 half 路径。

#### 3.2.2 InferShape 与校验

输入和输出 shape 规则：

```text
x:         [N,C], rank == 2, C > 0
target:    [N,K], rank == 2, target.N == x.N
is_target: [N,C]
y:         reduction == "none" ? [N] : [1]
```

InferShape 读取 reduction 属性并只接受精确字符串 `none`、`mean`、`sum`；属性缺省时按 `mean`。x/target 的已知 batch 维不相等时返回失败；静态 rank 非 2 或静态 `C<=0` 时返回失败。动态维度不在 InferShape 阶段做数值比较，留给 Tiling 的运行时 shape 校验。`N=0` 时 mean 的参考行为是 NumPy 空均值 NaN，题包没有声明是否支持空 batch；设计暂定要求 `N>0`，确认任务输入范围后再决定是拒绝还是定义空输入语义。`K=0` 可按无标签行处理，若框架不支持零维输入则需另行确认。

#### 3.2.3 Tiling 校验

Tiling 对实际 storage shape 再校验 rank、batch、`C>0`、dtype、reduction 和尺寸乘法溢出。获取 AIV 核数及 UB 容量；workspace 初版为 0。所有由 `N*C`、`N*K`、tile 字节数推导的整型运算采用 64 位，并在窄化到 API 字段前检查范围。

### 3.3 Tiling 设计

#### 3.3.1 工作划分

loss 依赖同一行中目标类和非目标类的配对关系。初版以样本行为基本所有权单位：一个 block 独占若干完整样本行，负责该行的 is_target 写回和 loss，避免跨核写同一行或对计数做原子累加。

- `reduction=none`：按 N 行在 AIV 核间分配，每个 block 独立写 `y[i]` 和 `is_target[i,:]`。
- `reduction=mean/sum`：题包是单 kernel 模板，且不同 block 间没有可依赖的全局 barrier。首版使用单 block 完成所有行及最终归约，保证 y 不存在未初始化累加或竞态。该路径是性能风险；若评测尺寸较大，需先确认运行框架支持受支持的两阶段归约后再设计多核 partial-sum 方案，不能用无序 atomic add 冒充归约正确性。
- 小 shape：当总工作量小于启动/同步成本时减少 block 数，首点 `[4,6]` 以端到端延迟为主，不期待靠增加并行度获益。

#### 3.3.2 UB 与 class tile

每行计算不分配完整 `[C,C]` 矩阵。按 UB 预算动态选择非目标类别方向的 `classTile`，逐 tile 读取 `x[i,j]`，构造计数/非目标 mask 和 hinge 向量并归约。target 前缀也按 UB 可容纳的块处理，扫描时保持从左到右的顺序，确保遇到第一个负值后忽略全部后续标签。

UB 预算至少包含：x 的 class tile、target tile、int32 计数 tile、非目标 mask、FP32 margin/乘积/归约临时量，以及 TPipe 对齐和预留空间。`classTile` 按所用向量指令的对齐粒度向下取整；末 tile 使用有效元素数 mask 或 DataCopyPad，所有输出只写有效字节。具体 buffer 数量及 tile 上限以 910B/CANN 实测和编译器资源报告校正，不在本草案虚构固定 UB 数值。

推荐的可实施基线：

1. 顺序确定每行有效前缀长度 `L_i`。
2. 对每个 class tile 初始化计数；扫描有效 target 前缀，将落在该 tile 范围的合法标签累加到对应计数，并写出 is_target。
3. 对前缀中的每个合法目标标签位置 `l`，加载 `x[i,l]`；对 class tile 中所有 `cnt[i,j]==0` 的类别，向量计算 `max(0, 1 + x[i,j] - x[i,l])` 并累加。
4. 重复标签按 target 位置重复执行，因此自然实现 `cnt[i,l]` 加权；全部 class tile 完成后除以 C，写出该样本 loss。

上述策略不物化 pairwise 矩阵，临时空间与 tile 成正比，计算量约为 `O(L_i*C)`，标签统计还需扫描 target 前缀。为避免大 K 下反复从 GM 读取 target，优先在 UB 容量允许时缓存 target 前缀；超过容量时按连续块扫描，并通过 profiler 判断数据搬运是否成为瓶颈。若后续改为“先建完整计数向量，再遍历目标类别”，需证明重复计数、target 截断和 mask 语义相同。

#### 3.3.3 Tiling Data 与 Key

TilingData 至少记录 `N`、`C`、`K`、每 block 起始/处理行数、class tile 长度、target tile 长度、reduction 类型和必要的尾块信息。字段使用 Host/Kernel 共用的固定宽度类型；变更后同步更新 Host UT 和 Kernel UT fixture。

tiling key 建议按 x dtype 分 3 路：FP16、FP32、BF16。reduction 作为 TilingData 小枚举传入，避免无必要地膨胀 key 数量。若后续目标芯片的 dtype 指令路径确实不同，再经测量决定是否增加 key 维度。

#### 3.3.4 Workspace

初版 workspace 为 0。`none` 的 y 每行独立写出；`mean/sum` 由单 block 生成最终标量，不执行跨 block 累加。只有当任务运行框架确认可在同一算子调用内合法调度两阶段 kernel 后，才评估每行/每 block partial loss workspace 与第二阶段归约；workspace 大小、生命周期和性能都需要纳入测试。

### 3.4 Kernel 设计

#### 3.4.1 行处理流程

```mermaid
flowchart TD
    A[读取一行 target] --> B[找首个负标签并确定有效前缀]
    B --> C[分块统计合法类别出现次数]
    C --> D[写出 int32 is_target 行]
    D --> E[按目标标签位置遍历]
    E --> F[分块计算非目标类 hinge 并归约]
    F --> G[除以类别数 C]
    G --> H{reduction}
    H -->|none| I[写 y[i]]
    H -->|mean 或 sum| J[累加样本 loss 并写 y[0]]
```

#### 3.4.2 数值路径

x 的 FP16/BF16/FP32 输入在计算中转换或保持为 FP32。margin、ReLU、重复次数加权和损失累加均采用 FP32 中间量，写 y 前转换回 x 的 dtype；`is_target` 使用 int32 累加和写回，不经过浮点转换。

参考实现的 pairwise margin 和乘法输入为 FP32，但 NumPy 对每行 `sum(dtype=float64)`，`mean/sum` 也经 float64 后再转换到 x dtype。Ascend C 初版计划使用 FP32 向量归约，归约顺序与 NumPy 不同；首测 float32 容差为 rtol/atol/tol=`1e-4`，尚未给出 FP16/BF16 或隐藏用例的容差。必须用目标机比较验证误差；若边界用例超差，先调整分段/树形归约，再评估代价更高的补偿累加方案。

比较与 hinge 路径应显式处理 `margin > 0` 的正值区域，等于 0 的项贡献 0。不得把目标类别只当二值标签，也不得把有效前缀外的标签纳入计算。NaN/Inf 输入按参考代码的 NumPy 运算传播规则处理；该行为和目标 TBE 的一致性目前没有随包提供原始证据，列为 NPU 复测项。

#### 3.4.3 尾块与同步

输入行和类别 tile 可能不满足 32B 对齐。GM 到 UB 的读写只访问有效范围，尾块根据 CANN 9.x 对应 API 的字节单位配置 DataCopyPad 或 mask；不允许假设补齐 lane 是有效输入。目标计数写回仅覆盖 `[0,C)`。跨 MTE/Vector 管线读写依照所用 API 明确同步，避免无依赖的全局同步；具体事件和 barrier 以 CANN 版本头文件及 NPU 编译结果为准。

### 3.5 性能分析与优化计划

该算子的主要计算为目标标签与非目标类别的 hinge 配对，复杂度随 `L*C` 增长；类别计数和 target 前缀扫描还带来额外内存访问。大 C 时不能分配完整 `C*C` 临时矩阵，否则 UB 无法容纳且写读流量过大。初版采用 class tile 控制空间，并按 N 分核处理 `none` 模式。

预期瓶颈和后续方向：

1. 首点 `[4,6]` 很小，kernel 启动与 ACLNN 调度可能主导；避免为了并行而增加额外 kernel。
2. 大 N、`none` 模式可按样本行占满 AIV 核，逐项记录 target 扫描、pairwise 计算和输出时间。
3. 大 C、较短 target 前缀时，计算复杂度取决于有效目标数 L；不对不存在的目标类执行 hinge 运算。
4. `mean/sum` 的单 block 正确性路径可能限制吞吐。只有确认支持合法的两阶段调度后，才采用多核部分和；在此之前需与内置 TBE 逐形状比较，不能宣称性能达标。
5. 以同设备、同 CANN、同输入和同计时方法对比内置 TBE。题包没有提供正式倍率门槛，设计阶段不填造速度数字或 PASS 结论。

## 四、支持硬件

| 芯片 | 题包配置 | 本设计状态 |
| --- | --- | --- |
| Ascend 910B | OpDef `AddConfig("ascend910b")`；README 示例环境 CANN 9.0.0/9.1.0 | 目标平台；尚未在本机编译或实测 |
| Ascend 910_93、Ascend 950 | CMake 模板列出可选 compute unit | OpDef 未注册，且本设计未验证；不声明支持 |

## 五、算子约束限制

1. x 和 target 必须是二维 ND Tensor，batch 维相同；C 必须大于 0。
2. target 标签小于 0 时从该位置起截断；有效前缀内大于等于 C 的正标签忽略。
3. target 标签允许重复；`is_target` 记录精确次数，loss 按次数加权。
4. `reduction` 只接受 `none`、`mean`、`sum`，缺省为 `mean`。
5. y dtype 与 x 一致；`is_target` 固定为 int32。
6. 初版要求 N>0；空 batch 和运行时动态维度的支持需由正式题目输入范围确认。
7. 输入/输出按 ND 连续访问；非连续场景由 OpDef `AutoContiguous` 行为保证，需在外层接口测试确认。
8. 每类计数使用 int32，要求 K 不超过 int32 可表示的出现次数上限；真实张量尺寸通常远低于该上限。

## 六、特性交叉分析

- **重复标签与 loss 归约**：重复标签既增加 `is_target` 计数，也线性增加该目标类对应的 hinge 项；二者必须来自同一有效前缀。
- **target padding 与越界标签**：负数是终止符；越界正数只是无效类别，不终止扫描。混淆两者会同时污染计数和 loss。
- **reduction 与输出 shape**：`none` 返回 N 个逐样本值；mean/sum 返回长度为 1 的 Tensor。仅返回标量值但 shape 为 `[]` 会被样例 verifier 拒绝。
- **低精度与计数输出**：x/y 的 FP16、BF16 只影响损失数据路径；target 和 is_target 仍是 int32。BF16 必须实例化 `bfloat16_t`。
- **批次归约与并行度**：样本 loss 可独立计算，但 mean/sum 需要跨 N 汇总。单 kernel 多 block 若无全局同步，不能直接共享累加器；初版采用单 block 归约，后续的并行化需有合法的两阶段执行方案。

## 七、可维可测分析

### 7.1 精度与性能标准

| 类别 | 标准 | 来源/状态 |
| --- | --- | --- |
| 输出 shape/dtype | 与 `case.json` 和 reduction 契约完全一致；y 长度为 1 或 N，is_target 为 int32 `[N,C]` | 题包明确 |
| is_target 精度 | 所有 int32 元素逐项完全相等 | 首测 `sample.py` 的 rtol/atol/tol 均为 0 |
| y 精度 | 首测 float32 使用 rtol=atol=1e-4，允许错误元素占比 tol=1e-4；单元素输出实际要求通过该元素比较 | `case.json` / `sample.py` |
| 低精度及隐藏用例 | 以正式任务要求为准；题包未提供完整容差表，必须补测 | 待确认 |
| 性能 | 在 910B/CANN 指定版本上逐用例记录 AICore/端到端时间，并对比同环境内置 TBE | 题包未给具体倍率或绝对门槛；当前无实测 |

### 7.2 测试矩阵

| ID | 类别 | 输入/属性 | 验证点 |
| --- | --- | --- | --- |
| F01 | 题包首测 | x `[4,6]` FP32；target `[4,6]` int32；mean | y=`[1]`、is_target=`[4,6]`，与 reference 比较 |
| F02 | reduction | 同一输入分别 none/mean/sum | y shape和值符合三种定义；none 为 `[N]` |
| F03 | dtype | FP16、FP32、BF16 | y dtype 与 x 一致；BF16 不走 half 位型解释 |
| F04 | 标签计数 | 重复标签 `[3,2,0,3,0,0]` | 类别 3 计 2、类别 2 计 1、类别 0 计 3 |
| F05 | 首负数截断 | `[3,3,-1,2,2]` | 类别 3 计 2；负数后标签不参与 |
| F06 | 越界正标签 | C=4，target `[5,1,4,2]` | 5/4 忽略但不截断；类别 1/2 各计 1 |
| F07 | 无正目标 | target 前缀为空或首项为负数 | is_target 全 0；逐样本 loss 为 0 |
| F08 | 全类别均为目标 | 前缀覆盖所有类别 | 没有非目标类，loss 为 0；重复次数仍正确 |
| F09 | hinge 边界 | 令 `1+x[j]-x[l]` 小于、等于、大于 0 | 分别贡献 0、0、正 margin |
| F10 | K 与 C 不同 | `[N,C]` 与 `[N,K]` 且 N 相同 | 按 reference 前缀规则处理，输出 shape 不变 |
| F11 | 大类别数 | 例如 C=4096，含稀疏、重复目标 | 无 C×C UB 临时矩阵；尾块正确；记录耗时 |
| Q01 | 非法 rank | x 或 target 非二维 | InferShape/Tiling 明确失败 |
| Q02 | batch 不匹配 | x.N != target.N | 明确失败，不越界读写 |
| Q03 | 非法 reduction | 不在 none/mean/sum 内的字符串 | 明确失败 |
| B01 | 浮点特殊值 | x 含 NaN、+/-Inf | 与任务规定/目标 TBE 对齐；当前列为待实测语义 |
| B02 | tile 尾块 | C、K 不满足 32B/向量对齐 | 不越界，补齐 lane 不污染有效结果 |
| P01 | 性能 | 小 `[4,6]`、中等尺寸、大 N/大 C，各 reduction | 同环境与 TBE 比较并保存原始日志 |

### 7.3 验证方法与记录

1. Host UT 覆盖合法/非法 shape、reduction、三种 dtype 的 tiling key、TilingData、blockDim 和 workspace；删除当前占位期望值并按最终实现更新。
2. Kernel UT 覆盖重复标签、截断、越界标签、hinge 边界和 class/target 尾块；fixture 必须与最终 TilingData 字段一致。
3. 在配置好的 CANN 9.0/9.1 及 910B 环境执行 `bash build.sh -j4 -e`，通过题包 sample.py 检查文件大小、dtype、shape 和数值。CPU UT 不能替代 NPU 端到端校验。
4. reduction 三种属性、FP16/BF16 及边界语义需要自建输入，因为题包 `samples` 只包含首个 `[4,6]` mean 样例。
5. 性能日志记录芯片、驱动/CANN 版本、shape、dtype、target 前缀分布、reduction、基线和逐次耗时；无实测结果时只报告待测，不写通过。

## 八、兼容性分析

这是新建自定义算子，不修改系统内置算子，不涉及已有 ABI 迁移。自定义 OpType 为 `MultilabelMarginLoss`，和内置算子同名；题包通过 `multilabel_margin_loss_custom` 包及 `ASCEND_CUSTOM_OPP_PATH` 加载。评测环境的自定义算子优先级和内置实现选择方式需要按题包运行脚本验证。输入/输出的 public ACLNN 参数顺序以生成的 `aclnn_multilabel_margin_loss.h` 为准，不自行更名。

## 九、待确认事项

1. 正式 problem 2327 任务说明、隐藏测试的 shape/dtype/reduction 组合及目标硬件/CANN 版本。
2. 参考实现注释记录的“首个负标签截断、重复次数计数”是否仍与当前官方评分基准完全一致；需要保存对应 910B 原始比较结果。
3. 对 x 中 NaN/Inf 的规定，以及 FP16/BF16 y 的比较容差。
4. `N=0`、`K=0`、运行时动态 shape 和 K 与 C 不相等是否属于评分输入范围。
5. reduction mean/sum 是否允许在一次算子调用内使用两阶段 kernel；若不允许，单 block 方案的大输入性能是否满足正式门槛。
6. 性能正式门槛及评分计时口径。题包当前只提供首测样例，没有基线时间或倍率要求。

