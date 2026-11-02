# 分支用 Compare+Select，禁止掩码乘法

Golden 是 `np.where`；`zero_one * (x*g)` 在 `g=±Inf` 等会变 NaN，与 golden 不一致。  
Inf 场景必须 Select，不能 TBE 0/1 公式落地。详见本题 `design.md` §2.1。
