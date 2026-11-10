# J050: upgraded environment and baseline calibration

Verified 2026-11-05. CANN install path resolves to cann-9.3.0, install version 9.3.0; compiler/OPP report 9.3.0-beta.1. msopprof 26.3.0. Driver 25.5.5, runtime SoC Ascend910_9362, build target ascend910_93. Runtime linkage was inspected after sourcing the new environment; CANN libraries resolve to 9.3.0. No performance delta against 9.0 is inferred from these new measurements.

J050 source archive is byte-identical to J035: SHA256 212183985a1795c76ca2fddf298d53ca4229321cdcdaa87517882b8ae17bbd35. New build, package, private install and precision campaign all exit 0. Three-dtype finite regression 780/780; 57 stress cases, 5184 invocations with both outputs checked every invocation, pass.

## A/A calibration

Nine cases: N/C/S = 127/8192/1, 81/8256/1, 127/32/1, each FP32/FP16/BF16. A and B are labels for the same J050 binary. Four rounds, ABBA/BAAB, shuffled case order; five device allocation addresses identical within each process for each case.

- Ordinary profiling: 144 windows, 350 kernel calls each; discard first 200, use median of last 150. 50400 kernel records. Maximum absolute A/B-label delta 0.0195 us, 95th percentile 0.0100 us; all nine cases below max(0.1 us, 2%).
- No-profiler enqueue: two separate processes, 144 windows each; warm 10240 kernels/window, 24 event samples of 128 kernels, discard first four samples. Maximum false delta 0.466992 us, 95th percentile 0.440351 us; only 10/18 case-trials below gate. Between-process average shifts range -0.3637 to +0.3300 us.
- All outputs matched reference. These measurements validate only the stated protocol, not official scoring or arbitrary cross-process comparisons. AIV pipeline activity durations can overlap and are not additive critical-path times.

Conclusion: continuous-enqueue timing alone cannot support sub-microsecond candidate claims. Subsequent A/B/C needs repeated baseline controls and balanced order. Same-process A/A stability does not prove cross-process A/B stability. No conditional dispatch rule has yet been accepted.

Cloud raw evidence: /home/developer/2329-JadeYara-20261104-1812/J050-layout-campaign and J050-aa. Local evidence/ contains environment, completion logs, 432-window summary and analysis. Full raw archive has been retrieved and SHA/CRC checked: E:\suanzi\cann-ascendc-ops-collab\tasks\active\PReluGradUpdate_2329\explorations\logs\EXP_J050_J052_CANN93_ABC_20261105\J050-J052-910C-evidence.zip. See J052/RESULTS.md and J052-verification.json.
