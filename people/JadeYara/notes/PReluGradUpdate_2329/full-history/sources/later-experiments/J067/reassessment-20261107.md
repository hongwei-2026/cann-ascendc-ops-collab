# Reassessment: official feedback, actual TBE contract, unexplored structures

## Official feedback, not inferred hidden shapes
Source: user screenshot libfile_b88fd11212a4819197d08c6d1563d2b8, Library OCR. J065 allfive Pass /0.00%; measured7.94/8.06/7.99/5.55/5.81us. Best-time column3.79/3.75/3.76/3.71/3.75us is not this submission.
Historical user-reported J035:7.15/7.29/7.11/5.88/6.06us; J056:7.21/7.16/7.19/5.58/5.84us. J065 firstthree roughly10.6–12.4% slower vsJ035, lasttwo roughly4.1–5.6% faster. These are different official submissions, not a controlled same-process experiment. J063 only has user's unchanged-score report, no independently read exact timing. No shape or dtype is inferred from point times. Do not promote J065 globally based solely on cloud45-input speedups.

## Actual installed TBE recovered
reference-evidence/prelu_grad_update.py is the unmodified CANN9.3 installed ops_legacy/dynamic implementation, Apache2.0 header retained, SHA256 c97cc823da4208f11f59a491231149c0b81105f6740bed9a707c28f555f9f9cf. This version need not be the Judge's version.
- Lines58–76/313–317 select ND/NCHW and conditional NC1HWC0; no NHWC declared.
- Lines296–304 allow FP32/FP16/BF16 and require equal three-input dtype. BF16 format selection depends on bf162f32 support.
- Lines191–196 broadcast grads/features, rather than require identical shapes.
- Lines235–237 support weights rank=features rank-1 (except1), inserting leading1. Thus ND[2,3,4] with weights[3,4] is spatial weights, not merely a renamed channel vector. Lines240–244 handle blocked NC1HWC0 weights.
- Lines223–229 handle unknown rank. Source documentation says grads/features do not support1D.
Current engine only registers ND, rejects differing input shapes and non-rank1weights, and treats storage dim1 asC. Existing970-group success covers that implemented subset, not all TBE formats/broadcasting.
check_shape_extract.py records exact same-version validation. Rank<=8 is actually checked; static/dynamic checks do not reject zero dimensions. However max_size2^31-1 is an unused default, not an enforced limit. Zero-dimension classifier/DSL execution remains unconfirmed; do not call the entire empty-shape contract proved.

## Reference vsTBE numerical conflict
The TBE after_v200 explicit DAG uses0/1 coefficients (122–127), then products (199–200), quantizes dx (202–207), and adds dx_quantized*0 into update (211–212). Current template uses where and does not include that propagation.
Under IEEE interpretation, finitehalf x=-1,g65504,w2 yields TBE-DAG dxInf/updateNaN, while template gives dxInf/update-65504. FP32 maxfinite*2 has analogous behavior. Nonfinite positive x/weight/grads also differ. These are source-DAG/CPU-derived examples and now confirmed by the compiled builtin public-ACL probe on the same910C. Do not confuse passing the current where-golden with full TBE-DAG equivalence. Neither the mere1e-3 threshold nor five-point Pass resolves this contract conflict.

## Evidence-ranked next structures
1. Native FP16 arithmetic+comparison+selection, independently on J035, no TPipe/ShortS combination. P030 promotion was a debugging attempt, did not fix507035; P031 unaligned Duplicate did. So nativehalf was never independently performance-disproved. Finitehalf independent products fit exactly in FP32 (22significand bits), making RNE directhalf multiplication equivalent to FP32-product->RNEhalf if hardware modes agree. Original strict checker remains stronger than the requested numeric tolerance and is not relaxed to force a pass.
2. Continuous block broadcast for arbitraryC and mediumS, not whole-row ownership. Current oddC S1 and generalS clamp tile toC/S, causing many tiny transfers; ShortS only covers2..32. Batch weights then vectorized gather/repeat expansion can remove that structural discontinuity while retaining original global block partition. Avoid merely widening a magic bound without changing weight preparation.
3. BF16 retain FP32 arithmetic but explore16-bit payload selection after products; BF16 must not be treated as half arithmetic.
4. Per-core shared-weight/current-channel scalar cache, not the failed full FP32 long-buffer cache and not only P105 read reordering. First inspect actual machine code for compiler hoisting.
Before broadformat implementation, define normalized logical dimensions/broadcast strides/blockedC0 mapping. Keep fast ND path and a correct generalized mapping separate; never infer support from AutoContiguous alone.

## J067 current revision evidence
Attempt1 built but strict channel-edge found false selection at element2032; shared all65536half feature patterns passed. Official Compare count API requires calCount*sizeof(T) multiple256B, so old64-element FP32 alignment was invalid for half tails (400->448half=896B). Attempt2 changes half Compare/Select count and Init buffer alignment together to128; queues must also grow to avoid tileLen64 overflow. Mul/Muls/copies remain currentNum. Official rule: https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha001/API/ascendcopapi/atlasascendc_api_07_0066.html
Attempt1 full failure evidence retained with original sources,148files, SHA4cce25980f0a5899bef8a2bbc9a7777040cef83ca9b3f162f84fafb2f98869cf. No performance was started after that failure.
Attempt2 sixroutes * all65536feature bitpatterns, maxfinite/extreme g/w:12groups,ABC72 and standard24 strict calls passed. Original970groups,ABC4026 and standard1342 strict calls also passed. Full performance completed:94inputs/789600calls;25nativehalf inputs all150rounds win; controls559/564. FP32/BF16 math and Host/routing unchanged. No official submission, commit or push.

## Actual builtin TBE semantics confirmed on device
Public aclopCompileAndExecute executed PReluGradUpdate with ASCEND_CUSTOM_OPP_PATH unset, two64-element scalar-weight probes, FP16 and FP32. Both outputs matched the installed source DAG64/64 elements for each output; update differed fromwhere32/64 in each dtype. The finite-overflow example is therefore now actual device evidence, not only CPU interpretation. Scope remains these two cases, not the full TBE contract.
The probe first needed its public header path corrected; after both successful NPUcalls, its final Python heredoc had a CRLF postprocessing NameError. Binary output files and two success logs were independently validated without rerunning NPU. Errors preserved, script normalized for reproduction. tbe-contract-evidence.zip includes full source/inputs/outputs/logs and canonical validation; SHA7326e53da2cb780e1c610df2691eea7f90a5a6d3a0c1586cf14e77b8872bded0.
