# 开发日志 — PadV3Grad_2328

## 2026-11-07

- 按模板 `reference.py` 实现 Pad 反向：constant 取中间，edge/replicate/reflect/symmetric/circular 按周期下标累加。
- x 支持 fp16/fp32/bf16，paddings 支持 int32/int64。两种排布都在 host 折成 front/back。
- `local_semantic_check.py` PASS。本机无 NPU，未跑 `build.sh -e`。
