# PReluGradUpdate_2329：云端测试操作交接

## 1. 使用范围

本文件记录已有成功流程及可供接手人使用的复现命令。**本次交接没有执行这些新复现命令。**

下面分为：

- 已运行成功的环境、工具、脚本与参数。
- 为避免覆盖历史目录而改写的复现命令；这部分仅经过已有脚本逻辑对照，尚未在新目录端到端运行。

接手前先确定用户是否重新授权设备作业。不要把交接手册当作自动启动指令。

## 2. 已核实环境

| 项目 | 已有实测 |
|---|---|
| 云主机规格 | 用户提供1×910C、40 vCPU、240GiB |
| 运行时SoC | Ascend910_9362 |
| 编译目标 | ascend910_93 |
| AIV | 40 |
| 物理UB | 196608字节 |
| 预留本地内存 | 8192字节 |
| 驱动 | 25.5.5 |
| 旧CANN | `/home/developer/Ascend/cann-9.0.0` |
| 新CANN | `/home/developer/Ascend/cann-9.3.0` |
| 当前规范路径 | `/home/developer/Ascend/cann`，上次指向9.3.0 |
| 新安装版本 | 9.3.0，V100R001C25B114 |
| compiler/OPP | 9.3.0-beta.1 |
| msopprof | 26.3.0 |

旧9.0与新9.3源码相同也可能生成不同机器码。不要混用旧二进制和新环境测量。

依据：`J052/RESULTS.md:5–20`、J050环境日志、J052 ELF对比。

## 3. 连接与会话环境

使用当前VS Code/SSH已有主机别名，不更改代理、DNS、VPN、凭据或安全配置。

PowerShell示例，`<EXISTING_SSH_ALIAS>`需替换为当前已有别名：

```powershell
ssh -T -o BatchMode=yes -o ConnectTimeout=15 <EXISTING_SSH_ALIAS>
```

历史成功别名记录在本地`run_remote_j043.py`；AtomGit转发别名可能随实例变化，接手时以现有配置为准。不要另建未知账户或暴露SSH私钥。

进入远端后，历史成功会话环境为：

```bash
source /home/developer/Ascend/cann/set_env.sh

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}

ROOT=/home/developer/2329-JadeYara-20261104-1812
```

这些只影响当前会话；不需要修改shell启动文件。

先检查：

```bash
date -Is
readlink -f /home/developer/Ascend/cann
printf '%s\n' "$ASCEND_HOME_PATH" "$ASCEND_OPP_PATH"
command -v bisheng
command -v msprof
command -v msopprof
npu-smi info
```

运行时SoC查询采用已有成功方法：

```bash
python3 - <<'PY'
import ctypes
lib = ctypes.CDLL('libascendcl.so')
lib.aclrtGetSocName.restype = ctypes.c_char_p
assert lib.aclInit(None) == 0
print(lib.aclrtGetSocName().decode())
assert lib.aclFinalize() == 0
PY
```

更完整的只读环境核验脚本：

```text
inspect_j050_environment.sh
inspect_j050_build_environment.sh
```

在本机辅助目录保留。它们记录安装版本、实际目录、工具路径、组件版本和相关旧作业状态。

## 4. 开工前确认断点，避免重复作业

历史作业采用：

```text
<job>.pid
<job>.exit
logs/<job>.log
```

例如：

```bash
for job in J050-layout J050-aa J051-layout J052-layout J052-abc J053-build; do
  printf '\nJOB %s\n' "$job"
  test ! -f "$ROOT/$job.exit" || cat "$ROOT/$job.exit"
  test ! -f "$ROOT/$job.pid" || ps -p "$(cat "$ROOT/$job.pid")" -o pid,etime,comm
  test ! -f "$ROOT/logs/$job.log" || tail -n 12 "$ROOT/logs/$job.log"
done
```

这是供接手人执行的查询示例，本交接没有再次运行。

判断规则：

- `.exit=0`：已完成，不因终端断连重复运行。
- 有PID无exit：核对进程身份、日志时间和实际输出。
- PID不存在也不能直接等同成功，检查exit与日志。
- J053本轮明确查到无任何`J053*`路径；不要假设它正在云端编译。
- 不用`pkill python`、`killall`等宽泛命令处理用户或他人任务。

## 5. 原始成功脚本

本机辅助目录：

```text
C:\Users\lin\Documents\Codex\2026-11-04\task
```

主要文件：

```text
remote_build_j050.sh
J050-layout.sh
remote_j050_precision.py
setup_j050_stress.py
J050-aa.sh
remote_j050_aa.py
j050_timing_probe.cpp
J050-AA-CMakeLists.txt
J052-abc.sh
remote_j052_abc.py
analyze_j050_aa.py
analyze_j052_abc.py
analyze_j052_elf.py
```

对应云端文件和原始执行记录随J050–J052归档保存。

**不要原样重跑整套旧launch脚本。**其中使用`mkdir()`且目录名固定，还引用旧实验输入和前序exit检查。原样运行可能因目录已存在停止，也可能混入旧结果。应使用新的输出目录，保留原始脚本作为协议依据。

## 6. J035重新构建：新目录复现示例

以下命令依据`remote_build_j050.sh`改为新目录；**新目录复现本身未执行**。

`RUN`需选择个人目录下尚不存在的路径：

```bash
set -e
set -o pipefail

ROOT=/home/developer/2329-JadeYara-20261104-1812
RUN="$ROOT/handoff-repro-REPLACE_WITH_UNIQUE_TAG"
export ROOT RUN

mkdir "$RUN"
mkdir "$RUN/logs"

python3 - <<'PY'
import os, hashlib, zipfile
from pathlib import Path

root = Path(os.environ['ROOT'])
run = Path(os.environ['RUN'])
src = root / 'J050-code.zip'  # J050源码即冻结J035
expected = '212183985a1795c76ca2fddf298d53ca4229321cdcdaa87517882b8ae17bbd35'
assert hashlib.sha256(src.read_bytes()).hexdigest() == expected
with zipfile.ZipFile(src) as z:
    assert z.testzip() is None
    z.extractall(run / 'J035')
PY

cmake -S "$RUN/J035/code" -B "$RUN/native-build" \
  -DASCEND_COMPUTE_UNIT=ascend910_93 \
  -DCANNJUDGE_COMPILE_JOBS=4 \
  -DCMAKE_INSTALL_PREFIX="$RUN/install" \
  2>&1 | tee "$RUN/logs/configure.log"

for target in all binary package; do
  cmake --build "$RUN/native-build" --target "$target" --parallel 4 \
    2>&1 | tee "$RUN/logs/build-$target.log"
done
```

历史成功脚本还执行了`install`目标。上述复现首先使用构建目录中的vendor包；若确有需要再执行install，先确认prefix仍为个人RUN目录：

```bash
cmake -LA -N "$RUN/native-build" | grep CMAKE_INSTALL_PREFIX
```

不要安装到系统CANN目录，不覆盖整树系统头文件。

历史vendor包路径：

```text
<native-build>/packages/vendors/p_relu_grad_update_custom
```

确认本次实际产物：

```bash
OPP="$RUN/native-build/packages/vendors/p_relu_grad_update_custom"
test -d "$OPP/op_api/lib"

export ASCEND_CUSTOM_OPP_PATH="$OPP"
export LD_LIBRARY_PATH="$OPP/op_api/lib:$LD_LIBRARY_PATH"
```

J035目标不是脚本默认的`ascend910b`；必须保留`ascend910_93`。若现场工具链改变，重新记录版本，不直接复用9.3结论。

## 7. 测试runner构建

以下来源目录来自已成功完成的J050流程，接手时先确认仍存在：

```text
$ROOT/scale-batch-src
$ROOT/J021-control-src
$ROOT/J050-stress-src
$ROOT/J050-aa-src
```

它们不是J035源码ZIP自带的全部文件。换机器时需从完整证据归档及其依赖中恢复；只有源码ZIP不能完整复现历史测试。

新目录构建示例：

```bash
cmake -S "$ROOT/scale-batch-src" -B "$RUN/batch-build" \
  -DCODE_ROOT="$RUN/J035/code" -DOPP_ROOT="$OPP"
cmake --build "$RUN/batch-build" --parallel 4

cmake -S "$ROOT/J021-control-src" -B "$RUN/profile-build" \
  -DCODE_ROOT="$RUN/J035/code" -DOPP_ROOT="$OPP"
cmake --build "$RUN/profile-build" --parallel 4

cmake -S "$ROOT/J050-stress-src" -B "$RUN/stress-build" \
  -DCODE_ROOT="$RUN/J035/code" -DOPP_ROOT="$OPP"
cmake --build "$RUN/stress-build" --parallel 4

cmake -S "$ROOT/J050-aa-src" -B "$RUN/aa-build" \
  -DCODE_ROOT="$RUN/J035/code" -DOPP_ROOT="$OPP"
cmake --build "$RUN/aa-build" --parallel 4
```

源目录与参数有历史成功依据；这里的新输出路径组合尚未运行验证。

## 8. 精度与压力复现

历史J050输出：

```text
$ROOT/J050-layout-campaign/precision
$ROOT/J050-layout-campaign/stress
```

其中`plan.txt`是实际runner参数清单。行格式：

```text
input_root output_root dtype N,C,S weight_len [iterations extra extra]
```

历史成功调用：

```bash
# 普通精度，一次调用；原参数形式
device_probe PLAN 1 0

# 压力runner，plan中含64/128次迭代
device_probe PLAN reuse 0
```

推荐从旧plan生成新输出路径，输入和golden保留原绝对路径。以下仅改写输出目的地：

```bash
python3 - <<'PY'
import os
from pathlib import Path

root = Path(os.environ['ROOT'])
run = Path(os.environ['RUN'])
pairs = [
    (root/'J050-layout-campaign/precision/plan.txt', run/'precision'),
    (root/'J050-layout-campaign/stress/plan.txt', run/'stress'),
]
for src, dst in pairs:
    dst.mkdir()
    rows = []
    for i, line in enumerate(src.read_text().splitlines()):
        fields = line.split()
        out = dst/f'case-{i:04d}'
        (out/'output').mkdir(parents=True)
        fields[1] = str(out)
        rows.append(' '.join(fields))
    (dst/'plan.txt').write_text('\n'.join(rows)+'\n')
PY
```

执行示例：

```bash
"$RUN/batch-build/device_probe" "$RUN/precision/plan.txt" 1 0 \
  > "$RUN/logs/precision.log" 2>&1

"$RUN/stress-build/device_probe" "$RUN/stress/plan.txt" reuse 0 \
  > "$RUN/logs/stress.log" 2>&1
```

压力runner来自`setup_j050_stress.py`，每次调用前预填输出，每次同步后检查两个输出；检查允许正负零等价，其他有限值按位比较。

普通精度必须进一步对照golden，不能仅凭进程退出0声称通过。可参考原`remote_j050_precision.py`中的`compare()`，读取：

```text
output_root/output/dx.bin
output_root/output/update.bin
input_root/golden/dx.bin
input_root/golden/update.bin
```

dtype映射在：

```text
$ROOT/inputs/J002/diagnose.py
```

该文件及NumPy/BF16支持是既有依赖；缺失时先恢复归档，不临时把BF16当FP16读取。

本轮历史验收数量：

```text
780精度案例
57压力案例
5184次压力调用
每次检查2个输出
```

迁移后必须核对manifest、输入SHA、实际条目数量与输出数。测试范围是有限值、零值、广播和边界，不宣称覆盖全部IEEE特殊值。

## 9. 同口径性能与A/A校准

### 普通profiling：已有成功命令形式

```bash
msprof \
  --output=NEW_PROFILE_DIRECTORY \
  --aic-metrics=PipeUtilization \
  --task-time=on \
  "$RUN/profile-build/device_probe" PLAN reuse 0
```

A/A历史plan：

```text
$ROOT/J050-aa/profile-trial1/plan.txt
```

可用上一节相同方法，仅将每行output_root改为RUN下新目录。不要写入原历史输出。

协议：

- 9个输入，三dtype。
- 每输入同一进程内固定五个设备地址。
- 四轮ABBA/BAAB；A/B实际是同一实现。
- 每窗口350次；丢弃前200次，后150次取Task Duration中位数。
- 总144窗口、50,400条内核记录。
- 从runner日志核验同一输入五个地址一致。
- 按CSV中的shape/dtype/op type核对条目，不仅按文件顺序猜归属。

历史最大A/A标签差0.0195μs；这是该轮观测，不是所有机器和未来运行的噪声保证。

### 无profiler连续enqueue：已有成功命令形式

```bash
"$RUN/aa-build/device_probe" \
  PLAN enqueue 128 24 80 0 OUTPUT_TIMING_CSV
```

来源：`remote_j050_aa.py`。

- 每event批次128次。
- 24个计时批次，分析时舍弃前4个。
- event_us除128得到连续提交间隔。
- 另存host_us。
- 不是隔离kernel执行时长，也不是官方比赛时间。
- 历史两进程A/A最大标签差0.466992μs，不能单凭它判定很小收益。

### 多实现对照的已知缺口

J052使用六轮平衡顺序：

```text
ABCA / CAAB / BACA / ACBA / BAAC / CABA
```

每轮四个独立进程；27输入；每窗口350次，丢前200。轮内五个可见虚拟地址一致，但跨进程A/A仍有最大1.041μs差异。

**真正同进程多实现固定分配对照尚未完成。**J053准备了三个测试算子名，但没有云端构建和机器码核验，不能直接运行并声称等价。

不要用可见7个GM参数猜测直调ABI。先前元数据`opParaSize=88`与可见参数数目不直接吻合，隐藏包装参数未核实；没有进行未知ABI设备调用。

## 10. profiler结果解释

普通PipeUtilization保留：

```text
Task Duration(us)
aiv_time(us)
aiv_scalar_time(us)
aiv_vec_time(us)
aiv_mte2_time(us)
aiv_mte3_time(us)
Block Num
Input Shapes
Input Data Types
```

- Scalar、Vector、MTE2、MTE3活跃时间可以重叠。
- 活跃占比不是独占耗时百分比，不能相加或相减获得关键路径。
- MemoryDetail/replay采样中的Task不替代普通性能对照。
- kernel replay与application replay的缓存行为不同，应保存实际参数。
- 本机910C/A3的核内上板Pipe/InstrTimeline能力受限；模拟器只作依赖定位，不作实板耗时。
- 已有实板MetricsProf局部窗口证据，不等同于完整指令timeline。
- 若工具升级，先看现场`msprof op --help`或`msopprof --help`，不要硬套其他版本参数大小写。

## 11. 日志和产物应保存什么

每次复现单独RUN目录，至少保留：

```text
环境版本与工具路径
源码ZIP和SHA
源码逐文件SHA
CMakeCache及完整构建日志
vendor包和runner指纹
输入manifest、随机种子、shape、dtype、权重长度
plan.txt、execution.json或等价命令记录
runner.log和退出码
精度comparison
原始profiling目录、op_summary CSV
每轮中位数、范围、配对差、A/A控制
开始与结束时间
```

不只保留最优值或截图。不把不同环境、预热、顺序、replay、enqueue与Task口径混成一张排名表。

## 12. 归档与常见故障

完整J050–J052归档：

```text
云端：
/home/developer/2329-JadeYara-20261104-1812/J050-J052-910C-evidence.zip

本地：
E:\suanzi\cann-ascendc-ops-collab\tasks\active\
PReluGradUpdate_2329\explorations\logs\
EXP_J050_J052_CANN93_ABC_20261105\
J050-J052-910C-evidence.zip
```

SHA256：

```text
ec5032a50c72932b59039a3216c814bf1715413e047f4dffd0efd9160e50e4ac
```

大小459,137,010字节，已完整取回校验。

| 现象 | 已有经验 |
|---|---|
| npu-smi首次报库加载失败 | 加载既有driver库会话路径后成功；不要先改系统配置 |
| 编译看似修改无效 | 对比实际构建目录与冻结源码SHA，历史发生过旧副本构建 |
| BF16 Cast后端失败 | 先确认target和实际工具链；不要混用旧910B结论 |
| 507035 | 曾由未对齐UB Duplicate引起；错误码本身不是唯一根因 |
| Windows执行环境断开 | 不等于云端作业失败，先查pid/exit/log |
| 大文件传输Bad packet length/Connection corrupted | 历史保留已取得块，以1MiB分块续传后全量SHA一致 |
| 目录已存在 | 旧脚本固定mkdir，使用新的RUN；不要删除历史结果绕过 |
| 相同代码计时翻转 | 保留A/A、未改路径控制和轮次；不只看均值 |
| 源码PASS但官方变慢 | 正确性与性能分开验收；本地shape不等于官方点 |

没有必要为了复现修改VPN、DNS、代理、网络安全配置或系统权限。网络路径问题与算子性能问题分别记录。

## 13. 当前停止点

- J050–J052构建、精度、计时及归档均已完成，退出0。
- J053只有本地准备，云端未启动。
- 用户已要求交接并停止新增优化。
- 下一位接手人先阅读源码对照文件和现有报告，再由用户决定是否重新启动实验。
- 平台提交、恢复成绩和查看官方结果始终由用户本人操作。
