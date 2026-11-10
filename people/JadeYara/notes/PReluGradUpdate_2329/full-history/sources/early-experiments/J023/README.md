

## J023 — 按编译实际 dtype 隔离模板函数体

J022预取只在FP16/BF16 legacy S=1执行，大形状回退路径没有执行该指令；已知其布局位置/字节哈希改变。当前每个输入dtype二进制包含全部三dtype共12个完整模板，原生编译命令定义DTYPE_GRADS=float/half/bfloat16_t。J023以J022为基线，在三个if constexpr分支增加std::is_same<DTYPE_GRADS,T>::value，裁掉当前二进制不使用的另外两种dtype函数体；Host/key/预取与数学不变。先检查三个ELF的实际体积/函数体，再498精度与J020/J022/J023冷热交替。保留J013/J018/J022。当前目标是减少代码布局耦合，不将体积缩小预设为延迟收益。
