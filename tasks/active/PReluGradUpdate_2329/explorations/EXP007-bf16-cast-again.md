# EXP007 — P019：源码已同步，bf16 Cast 再次炸 FATBIN

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-02 |
| 日志 | `explorations/logs/EXP_P019_20261102_175158/` |
| 结论 | verify **PASS**；编译失败根因 `not support bf16 type cast`（同 EXP002） |

## SOURCE 指纹

```text
repo_commit=7f27529...
source_kernel == build_kernel
CodeArts source verification: PASS
```

说明 P019 门禁有效：不再是旧 SyncFunc 副本。

## 编译错误

```text
fatal error: error in backend: not support bf16 type cast
ld.lld: cannot open ..._1.o / ..._2.o  (FATBIN 级联)
```

Claude 实现里对 `half`/`bf16` 统一 `Cast(...CAST_NONE/RINT)`；910B 拒 bf16 Cast → 该 tiling key 无 `.o` → fatbin 失败，整包 binary 目标挂掉。

## 处置（P020）

- bf16：位扩展 / RNE 截断逐元素，禁止 Cast  
- half：仍可用 Cast  
- 样例为 fp32，但 opc 会编三条 key，bf16 不过则整次 build 失败  
