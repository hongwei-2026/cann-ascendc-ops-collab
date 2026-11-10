

## J021 — 大输入后小输入回退的控制实验

不改算子。已有原始counter显示低精度tiny_after_resident J013 I-cache miss rate=0，J020约0.038/0.039，AIV cycles同步增加；FP32对照miss=0。cycles/aiv_time为聚合换算，不能当独立实时频率。旧驱动每case分配/释放设备内存，需隔离地址与顺序。J021仅新建诊断驱动，对J013/J020做fresh/reuse固定地址、前序resident/merged/scalar、25或250次前序、10ms间隔、同地址输入重写；tiny连续250次分早/中/晚窗口统计，四轮交替。已有dtype/输入/golden复用，无安装、无频率设置或安全修改。
