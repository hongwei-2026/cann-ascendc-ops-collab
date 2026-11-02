# 汇总 — PReluGradUpdate（合并定稿）

| 项 | 内容 |
|----|------|
| 状态 | **已定稿（可实现）** |
| Cursor 三路 | [AI-1_composer.md](AI-1_composer.md) · [AI-2_grok.md](AI-2_grok.md) · [AI-3_grok.md](AI-3_grok.md) |
| 本地 Codex/Claude | [AI-codex-claude_design_final.md](AI-codex-claude_design_final.md)（源自 `D:\华为算子比赛\PReluGradUpdate_problem_2329_plan`） |
| 实现主依据 | **以本地 design_final 为实施细则**；Cursor 三路作交叉校验与止损纪律 |
| 隔离 | 仅本题 |

---

## 共识（全部来源一致）

1. 语义：`dx = x>0 ? g : w*g`；`update = x>0 ? 0 : x*g`；通道维 dim1；update **不 reduce**  
2. bf16 **独立 tiling key**，禁止走 `half`  
3. workspace=0；带宽题；先正确性后性能；小 shape 不抠 us  
4. InferShape 必须校验；TilingData 须含 C/inner 等元数据  

## 采纳本地设计的关键加严（覆盖 Cursor 粗案）

| 点 | 裁决 | 理由 |
|----|------|------|
| 分支实现 | **必须 Compare + Select**，禁止掩码乘法 / TBE 0-1 公式落地 | Inf 场景：掩码乘法与 numpy golden 不一致（已实测） |
| 对标口径 | **以 `reference.py` / numpy `where` 为 golden** | 与内置 TBE 在 Inf 两处可不同；评测若改 TBE 需 DEC 重对齐 |
| 切分 | **模式 R（sample-major）优先 + 模式 C（大 S）**；权重向量模式 R **整核只展开一次** | 比单纯 ROW_SLICE 更贴合小 inner；为本题主优化 |
| weights | 支持 **`[1]` 与 `[C]`** | 本地设计明确；InferShape 须校验 |
| tiling key | 0=fp32，1=fp16，2=bf16（与本地表一致；实现时与模板 MODE 命名对齐即可） | |
| Select 极性 | 四元素 UT 锁定（B03） | 极性反了会静默错 |
| Cast | 半精度升 fp32；回写 `CAST_RINT`；bf16 golden 勿用截断 | |

## 保留 Cursor 三路的纪律

- 性能止损 / 小 shape 仿真豁免 / 禁止假全核调 `[4,6,8]`  
- 一题隔离、path-ledger、≥3 分析流程（本地设计计为 Codex+Claude 长文，与 Cursor 三路一并归档）  

## 废弃清单

- 掩码乘法实现分支  
- update 通道归约 / atomic  
- bf16→half  
- 权重 expand 到 GM workspace  
- 无极性 UT 就上多核优化  
- 拿首点当性能门禁  

## 执行顺序

| 级 | 内容 |
|----|------|
| P0 | 按 `AI-codex-claude_design_final.md` 实现 Host/Tiling/Kernel；B03+样例冒烟 |
| P1 | 全核大 M ≥ TBE 95%（任务书）；有界排查 |
| P2 | &lt;10us 仿真豁免 |

## 代码落点

- 协作仓：`tasks/active/PReluGradUpdate_2329/workdir/code/`  
- 云上：CodeArts `/mnt/workspace` 用 **git clone/pull** 同步（见 `SYNC_CODEARTS.md`）
