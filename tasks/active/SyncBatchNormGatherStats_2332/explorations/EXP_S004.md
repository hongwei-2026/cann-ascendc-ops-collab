# EXP_S004 — QuantFp16Vec 保 NaN（平台失败 + 回退）

## 假设

`QuantFp16Vec` 往返 Cast 吞掉 `count==1` 路径的 `new_variance` NaN/±Inf；S003 只覆盖最终 half Store。

## 方法

在 `QuantFp16Vec` 第二次 Cast 前标量扫非有限值回写；补 `S_V` / `S_MTE3`。

## 结果

| 环境 | 结果 |
|------|------|
| 云 `build.sh -j8 -e` | **PASS**（0/6）→ `logs/EXP_S004/` |
| 平台 50 点 | 仍 **Wrong Answer**（1–5 Pass/0%；隐藏失败，与 S002/S003 同型） |
| 可见点用时 | **~6.05–6.69μs**（S002 约 5.87–6.23μs）→ **更慢** |

## 结论

**废弃。** 隐藏 WA 不是「Cast 后全 tile 标量保 NaN」能修的；该补丁有明显性能税。  
S005 已回退 `QuantFp16Vec` / `StoreAsInputDtype` 至 S002 语义；**禁止重开本族补丁**。
