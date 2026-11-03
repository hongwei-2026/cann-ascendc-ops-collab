# EXP023: Reuse the Zero Buffer and Shorten the Vector Schedule

- Date: 2026-11-03
- Owner: hongwei-2026
- Candidate: P103, based on the restored P100 source
- Status: local source checks pass; CANN build, cloud sample, and platform timing are pending

## Hypothesis

P100 writes zero to `dxFloat`, uses that value to compare `featuresFloat`, then overwrites `dxFloat`. Later it writes zero to `featuresFloat` so `updateFloat` can select zero for positive inputs. Since the first zero buffer is still live until the update select, the second `Duplicate` is redundant. The independent `Muls`, `Mul`, and zero `Duplicate` also do not need an intervening vector barrier.

## Change

P103 keeps `zeroLocal` aliased to `dxFloat`, computes `updateFloat` before writing `dxFloat`, and uses `zeroLocal` as the positive branch of the update select. It issues one `PIPE_V` barrier after the independent `Muls`/`Mul`/`Duplicate` stage, then preserves barriers after `Compare`, the update select, and the dx select.

The float compute section changes from seven vector operations and six explicit `PIPE_V` barriers to six operations and four barriers. No allocation, tiling, copy, conversion, output rounding, or compare/select polarity changes.

## Dependency Check

The local checker extracts and interprets the actual C++ vector section. It canonicalizes `zeroLocal` to `dxFloat` and checks RAW, WAR, and WAW dependencies between operations. The required barriers remain across zero-to-compare, compare-to-select, update-read-to-dx-write, and other producer/consumer dependencies. The update select produces a separate output tensor before `dxFloat` is overwritten.

## Validation

- `compute_schedule_check.py` passed on P103 and the P100 source snapshot: P100 7 ops / 6 barriers; P103 6 ops / 4 barriers.
- The same check compared fp32 and fp16 outputs with `samples/reference.py` over lengths 1, 7, 8, 15, 16, 63, 64, 65, 127, 128, 257, and 8255. Inputs included NaN, infinities, signed zero, and subnormal values; scalar weights included zero, finite, infinite, and NaN values.
- `local_semantic_check.py` passed, including existing semantic, channel-boundary, and tiling checks.
- These CPU checks do not validate AscendC compilation, device behavior, or performance.

## Decision

Keep P103 as the local candidate for a CANN build and controlled platform measurement. Do not claim it exceeds P100 until the candidate source/build fingerprint is tied to a passing cloud run and same-environment timing. P100 remains the measured performance reference. No push was performed.

## Owner 复审（同日）

- **可试**：语义不变前提下少 1 次 Duplicate + 2 道 PIPE_V；update Select 前移并复用 `dxFloat` 零缓冲，依赖链有 barrier；本地 `compute_schedule_check` + `local_semantic_check` PASS。
- **风险**：仍是 B 类小 shape 微优化；真机可能无感或因调度差异变慢（P102 前车）。必须云测 0/192 后再同机相对 P100 采时。
- **下一闸**：CodeArts `-e`（日志进 `exp_logs/prelu2329/`）→ 平台 A/B；无 ≥2% 增益写 DEC 回退 P100。
