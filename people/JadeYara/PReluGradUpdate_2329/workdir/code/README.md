# PReluGradUpdate_2329：J077 复现与 UT 修复版

基线为官方提交 1221802；本修复版不修改 op_host/op_kernel。

已验证环境：CANN 9.3.0-beta.1、设备 Ascend910_9362、构建目标 ascend910_93。需要本地 GoogleTest 源码、nlohmann/json 头文件、SDK tikicpulib，以及 Python numpy。读取 bfloat16 数组时需要 ml_dtypes。

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}
export GTEST_SOURCE_DIR=/usr/src/googletest
export NLOHMANN_JSON_INCLUDE_DIR=/home/developer/opensource/json/include

bash build.sh --soc ascend910_93 -j4
bash build.sh --soc ascend910_93 -j4 -u
DEVICE_ID=0 bash build.sh --soc ascend910_93 -j4 -e
```

build.sh 默认目标仍为 ascend910b，可用 --soc 或 ASCEND_COMPUTE_UNIT 指定实际目标。切换目标会使构建指纹失效。当前 910C 上，910b-only 包会加载自定义 Host 却回退到内置 TBE 内核，造成 80 字节自定义 tiling 与内置 kernel 混用并报 507035；显式构建 ascend910_93 已通过。不要把修改包目录名作为通用跨型号适配方案。

-u 实际执行 7 项 Python 数值链测试、9 项 Host UT、1 项 float Kernel CPU 模拟器 UT，并校验模拟器的两个输出；不再使用旧模板的“默认跳过内层断言”说明。CPU 模拟器型号为 Ascend910B1，不等同于当前设备实测。-e 运行 [4,6,8]、通道权重 [6] 的 float 设备样例，并校验 dx/update；单个样例不代表完整输入覆盖。

samples/reference.py 是真实 TBE 算术顺序的 CPU 模型：

```text
neg = select(x > 0, 0, 1)
pos = select(x > 0, 1, 0)
dx = cast_output_dtype((pos + neg * w) * g)
update = cast_output_dtype((neg * x) * g + float32(dx) * 0)
```

不能用 where 跳过掩码乘法产生的异常值，也不能删除低精度 dx 量化和回读。update 逐元素、不归约。ND/NCHW/NC1HWC0 使用物理存储形状，不执行布局转换；权重按真实 TBE 分支次序 reshape 后只能广播到 grads/features 的输出形状。

J077 已核验：上述 UT 和设备样例通过；CPU 模型对 67 个已接受的真实 TBE 归档金标准为零逻辑差异，2 个被 TBE 拒绝的描述另记拒绝。完整日志、首次失败记录、目标包对照与哈希在上级实验目录 README.md 和 device-evidence.zip；CPU 模型不替代设备 oracle。
