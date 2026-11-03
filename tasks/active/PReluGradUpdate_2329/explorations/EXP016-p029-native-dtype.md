# EXP016 — P029：原生 half + bf16 位转换（治 Cast 污染）

| 字段 | 内容 |
|------|------|
| 日期 | 2026-11-03 |
| 前置 | EXP015 / P027–P028：同 TU 的 DtypeBridge/`ToFloat` 拖垮全部 key |
| 目标 | 三 dtype 能编过；平台 case4（507035）有低精度核可绑 |

## 根因（纠正）

不是「本机不能编 half」，而是 **`DtypeBridge` 显式特化 / `ToFloat`/`Cast` 会进入每一个 tiling key 的编译单元**，连 fp32 key（`40c7c86a`）也报 `not support bf16 type cast`。

## 改法

1. 删除 `DtypeBridge`；禁止 `Cast` / `ToFloat`
2. `float`/`half`：Compare+Select **原生 dtype**
3. `bf16`：仅 `if constexpr (is_same<bfloat16_t>)` 内位扩展到 fp32 再算（不实例化到其它 key）
4. pad/零值用位写 0，避免 `static_cast<half>(0)`

## 验收

CodeArts：`verify` PASS → `build.sh -j4 -e`；期望三 key Generating Done + 样例 PASS。失败则整包 log 归档 `EXP_P029_*`。

### 结果（EXP_P029_20261103_132332）

| 项 | 结果 |
|----|------|
| tip | `91f4596` |
| verify | PASS |
| keys | float `40c7c86a` / fp16 `7d1e7472` / bf16 `599588aa` **全部 Done** |
| 样例 | dx/update **0/192 PASS** |
