# 当前合同修订 — 2026-11-08 / J077

本节取代下方历史正文中关于 NumPy where、仅 [1]/[C] 权重及禁止 TBE 掩码乘法的旧裁决；历史正文保留以解释 J035 及早期实验，不再作为新实现的数值依据。

- 依据：CANN9.3 内置 prelu_grad_update.py（J067 归档）及 J068–J076 真实 TBE 设备 oracle。基线使用官方提交1221802下载源码；J035仅为历史回滚版本。
- neg=select(x>0,0,1)，pos=select(x>0,1,0)；dx_compute=(pos+neg*w)*g；先按输出dtype量化dx；update=(neg*x)*g+float32(quantized_dx)*0，最后按dtype量化update。低精度中间链按真实TBE/实际等价证明处理，不能把看似多余的乘法和舍入删掉。
- Inf/NaN及有限溢出时，以上不等价于旧where。逻辑比较区分有限值与NaN类别，±0等价；不声称NaN payload逐位相同。
- 输出shape为broadcast(grads,features)，weights只允许广播到该shape。权重reshape保持TBE分支次序和存储元素总数。ND/NCHW/物理NC1HWC0分别处理，rank上限8；未知rank及混合格式转换的支持边界按已验证证据记录。update逐元素、不归约。
- 本轮可写接续目录：people/JadeYara/notes/PReluGradUpdate_2329/experiments/J077。官方原包、历史实验和冻结生产不修改。
- 新 samples/reference.py 是CPU算术模型；真实NPU TBE输出仍是独立验收oracle，不能以CPU通过替代设备通过。

---

