# EXP001 — 模板空 Kernel 端到端基线（Wrong Answer）

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-02 |
| 作者 / AI | team + cloud CodeArts |
| 关联台账 | P007 / P008 |
| 阶段 | 环境 / 功能 |
| 结论 | **废弃作为实现**；保留为「未实现必 WA」基线证据 |

## 假设

题包模板可编译安装；空 Init/Process 跑 `-e` 应对不上 golden。

## 做了什么

```bash
cd /mnt/workspace/prelu2329/code
bash build.sh -j4                    # OK → custom_opp_ubuntu_aarch64.run
DEVICE_ID=0 bash build.sh -j4 -e     # 样例生成后 verify
```

环境：Ascend910B / CANN 9.1.0 / ubuntu aarch64（云端日志）

## 结果

- Build：**成功**
- Verify：
  - `Wrong Answer: dx.bin, mismatched=192/192`
  - `Wrong Answer: update.bin, mismatched=95/192`
  - `FAILED`
- 解释：Kernel/Tiling 仍为 TODO，输出非 golden；`update` 部分碰巧接近零/未写满导致并非 192/192。

## 结论与给后人的一句话

- **不要**在空实现上调性能或改对比阈值。  
- 下一闸：按 `design.md` / `AI-codex-claude_design_final.md` 实现 Host+Kernel 后再跑同一命令；对比本 EXP 的 mismatch 应明显下降至 0。  
