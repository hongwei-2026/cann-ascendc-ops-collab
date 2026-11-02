# 半错：极性 vs mask 失效（EXP003/004/P015）

样例 `[4,6,8]` seed=42：正 97 / 非正 95。

| 核上等效行为 | dx/update 错数 |
|--------------|----------------|
| 恒正支 `dx=g, upd=0` | **95 / 95** |
| 恒负支 `dx=w*g, upd=x*g` | **97 / 97** |
| Select 极性完全反转 | ≈ **192 / 192** |

**结论**：错数卡在 95 或 97 = **mask 未生效（Select 恒走一侧）**，不是「换一下 src0/src1 就好」。  
换 Select 参数只会在 95↔97 之间跳，根因仍在 Compare/mask 通路。  
证据：`tasks/active/PReluGradUpdate_2329/explorations/EXP004-mask-always-src1.md`。
