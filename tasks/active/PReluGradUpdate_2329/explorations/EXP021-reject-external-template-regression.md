# EXP021 — 拒收外部模板「Codex 再优化」（2026-11-03 15:03）

| 字段 | 内容 |
|------|------|
| 路径 | `D:\华为算子比赛\PReluGradUpdate_problem_2329_template\code\` |
| 触达 | 仅 `op_kernel/p_relu_grad_update.h` mtime 15:03；tiling 仍为 11-02 旧版 |
| 对照 | collab `people/hongwei-2026/...` tip `3bd65fb` / P100 |

## 结论：**回归，勿合入座位 / 勿交平台**

相对已冻结的 P031+P100，该 kernel 回退到早期失败形态：

| 点 | 外部模板 | 当前座位 |
|----|----------|----------|
| 同步 | `SyncFunc<HardEvent::S_V>` | `SetFlag`/`WaitFlag`（CANN 9.1） |
| mask | `CompareScalar` | `Duplicate(0)` + `Compare` |
| weight | 未对齐 `Duplicate(weightLocal[offset])` | P031 对齐 / P100 直接 `Muls` |
| 半精度 | 泛化 `Cast` + `ToFloat` | bf16 禁 Cast；half 限 `if constexpr` |
| Process | `progress += tileLen_` | P100：`+= currentNum` |
| tiling 同伴 | 小图单核；zip 仍见 16MB WS | P100：`WS=0` + 全核 |

## 处置

- **不**拷入 `people/hongwei-2026/`  
- 默认试交仍为 P100  
- 若 Codex 另有输出目录，再对照本表审一次  
