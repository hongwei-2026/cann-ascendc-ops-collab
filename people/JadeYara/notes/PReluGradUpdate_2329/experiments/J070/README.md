# J070: general compact-window broadcast fallback

Status: this device campaign is complete. Functional generalization passed the recorded cases; generic performance is not an overall improvement. See [final measured report](visible-agent/RESULT.md) and final standard source（本地归档：`visible-agent/attempt5/candidate-standard-source-attempt5.zip`，本次不上传 ZIP）. Production J035 is unchanged.

The sections below preserve the initial implementation record; their pending/not-run wording describes that earlier stage.

## Source and boundary

- `candidate/code/`: 51 source/build/test files, based on the on-disk J069 standard source.
- Five existing integration files changed: OpDef, InferShape, tiling, kernel entry, template keys. Three files added: shape utility, General metadata, General kernel. The other 43 baseline files are byte-identical.
- J069 itself remains 48/48 hash-identical. Its original Host selector body and every existing kernel remain unchanged. Positive same-shape ND + rank1 channel/scalar weights still call that selector and keep its old metadata and encoded key ordinals.
- `abc/code/`: generated A/C=frozen J069, B=mechanical rename of standard J070. New broadcast/layout cases cannot run through J069 A/C; use builtin TBE as their oracle and alias B to rule out silent builtin fallback.
- No production change, cloud upload/write/build, NPU operation, official submission, commit or push was performed by this implementation task. Strict SSH was used only to read exact CANN Host API/rank declarations.

## Actual contract

The saved CANN 9.3 source is `../J067/reference-evidence/prelu_grad_update.py`; precise function line ranges are in `reference-lines.json`.

- Compute derives output as broadcast(grads,features), then broadcasts the mapped weights to that output. Weights do not enlarge output.
- Weight mapping follows the source's exact branch order, including rank(features)-1 spatial weights and physical NC1 weights. The mapped storage count must equal the allocated weights count.
- Registered tuples mirror `op_select_format`: ND/ND/ND, NCHW/NCHW with ND weights, and NC1HWC0 for all tensors. Each tuple has equal input/output dtype. No extra NCHW rank4 assumption is made.
- NC1 is compact physical [N,C1,H,W,C0]. C19 has 32 weight lanes in storage, not 19. The kernel does not modulus by logical C or convert layouts.
- Known-rank symbolic dimensions are broadcast during InferShape; runtime dimensions must be concrete. The unknown-rank [-2] sentinel is propagated but graph compilation remains unverified.
- Rank8 is the actual TBE para_check limit, read from CANN 9.3 `python/site-packages/tbe/common/utils/para_check.py:43,651-653`. No invented rank2 minimum is imposed on grads; unusual scalar/rank1 combinations are source-based, not device-confirmed.
- Empty output uses a valid General key and one no-op block, before all divisor/suffix/GM access.

## General execution

New schMode values: 28/29/30. Encoded ordinals: full dtype build 18/19/20; fp16+float 12/13; float-only 6. The registration expressions are quoted SDK-supported `TILING_KEY_VAR` expressions. General metadata is a separate 760-byte struct, not a reinterpretation of J069 metadata.

Each core owns an output range with 32-byte-aligned global ownership boundaries. Within it, tile cap is fixed at 2048. A tile is additionally cut at the smallest output suffix period >=2048, so every source's exact hull fits the cap. Shared endpoint coordinates and O(rank) lexicographic boxes compute exact min/max, including broadcast carries. Nonidentity sources use one exact-byte contiguous GM DMA and local bounded int32 Gather. Identity sources use direct DMA/Cast. Scalar broadcasts use Gather index0. All absolute offsets, shapes and suffixes remain uint64; only proven bounded local differences enter int32. Tail indices clamp the ramp to n-1 before arithmetic. Source GM reads and output GM writes are never rounded past valid storage.

Arithmetic preserves the actual TBE DAG: neg=select(x>0,0,1), pos=1-neg; dx=(pos+neg*w)*g; update=(neg*x)*g+quantized_dx*0. FP16/BF16 inputs compute in FP32; dx is CAST_RINT and converted back before its zero term, then update is CAST_RINT. No where shortcut, native-half substitution, fused multiply-add or per-lane GM load is introduced.

UB at fixed tile2048: FP32 76288B; FP16/BF16 88576B. The 80% budget of 188416B is 150732B. Nine FP32 arrays, one byte/lane mask, two 256B phase pads; low precision adds one reusable raw input and two raw outputs. Compact input storage becomes neg after Gather. Index scratch reuses dx/pos before math. g/pos/neg phase A and x/w/dx/up phase A+256 separate critical two-source arithmetic reads. This is static accounting, not a measured bottleneck/performance claim.

## Completed local checks

`python validate_local.py` produced `local-validation.json` and `local-validation.log`:

- 192 randomized legal layouts, 1051 tiles, 2796864 lane indices; 57 windows have absolute source offsets >INT32_MAX. Included nested rank8, unit axes, carry boundaries, 2^32 resets and padded tails.
- All 17 previously measured TBE contract descriptors: 3480 physical positions, 34 outputs. The CPU input expansion and TBE math match the saved actual binary outputs, accepting differing NaN payloads. Corrected NC1 inputs/outputs come from `J068/device-evidence.zip:contract-r2/`; other contracts from `contract/`. This is reuse of existing device evidence, NOT execution of J070 on device.
- Targeted rejections: weights that enlarge g/x output, weight reshapes changing storage element count, and raw19 weights passed as physical32.
- Empty and known-rank dynamic inference checks, unchanged J069 source hashes, unchanged legacy Host body, retained key ordinals, separate new metadata, and exact UB allocation phases.
- `prepare_abc.py`: 42 mechanically renamed operator source files +4 build files, verified against their source mappings.

Initial CPU check failures are retained: attempt1 chose the known-invalid first NC1 descriptor outputs; attempt2 hit Windows default GBK decoding while reading source. The oracle selection and UTF-8 reads were corrected, no golden or acceptance threshold changed.

## Next gate, owned by parent

1. Transfer the final source manifest only after review. `device_build_prepared.sh` is prepared, NOT executed. It targets CANN9.3/ascend910_93, all/binary/package, standard and ABC, without install/test/submit.
2. Check all format/dtype registrations and General metadata ABI against compiled kernel metadata. Standard/B should have the same new kernels. Preserve old bodies where compiled output allows comparison; old source identity is not proof of identical performance.
3. Reuse all existing corrected TBE contract binaries and 1063 precision inputs on their applicable paths. Run new General shapes by custom alias B and standard directACL using proper origin+storage descriptors; verify actual selected kernel and no builtin fallback.
4. `proposed-device-cases.json`: 54 new candidate descriptors (18 shapes x3 dtypes), all unexecuted. Covers complementary g/x broadcasts, spatial weights, expansion relative to features but not output, P* crossing, tail, rank8/unit axes, NCHW and C19 NC1, N/C/S zero dimensions. C19 origin/storage metadata is explicit. Add NaN/Inf/signed-zero/finite-overflow patterns and low-precision all-bit feature patterns to representative General cases, not just finite random data.
5. Only after device correctness, measure a bounded same-process full-Task comparison with unchanged J069 routes as controls. No speed claim or universal TBE95% claim is made here. Unknown-rank compilation, mixed-format transformations, unusual scalar/rank1 forms and device General behavior remain unverified.

`proposed-contract-probes.json` adds three small FP32-only scalar/rank1 source-based probes. Query builtin acceptance first; rejection is contract evidence, not a candidate correctness failure. Existing empty_n and empty_c cases explicitly broadcast 0 against 1. No device allocation for >INT32_MAX offsets is planned.


> 2026-11-10 同步注：本页为原实验记录。仅公开选定文本证据，不表示所有相对路径产物均已上传；代码状态和下一步以[当前进度](../../progress-20261110.md)为准。
