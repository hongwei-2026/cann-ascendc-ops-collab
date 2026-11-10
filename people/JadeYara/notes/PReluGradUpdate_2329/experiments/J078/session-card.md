# J078 — Native FP16 unused scratch initialization

2026-11-08, JadeYara. Official1221802 baseline. Single variable: exclude native-half paths from four unused FP32 scratch-buffer InitBuffer groups in scalar/S1/merged/resident kernels. No arithmetic, Host, keys, tiling or synchronization changes. Source audit shows these four buffers are only read by the non-half compute branch. J054 was FP32 buffer aliasing and lifetime reuse; this is a distinct initialization-only hypothesis. J077 test/build fixes reused as tooling, not runtime optimization.

First gate: build and compare all 189 function bodies by dtype, format and key against official baseline compiled with identical CANN9.3 target910_93. If bodies are identical, reject as compiler-eliminated with no performance claim and no benchmark fishing. If they differ, perform actualTBE regression and fixed coverage A/A/B controls before considering promotion. No official submission.

Compiled gate: 174/189 bodies unchanged. FP16 scalar/S1/merged/resident shrink172–276B; splitC bytes change at same size (relocation differences preserved). Proceed with full inherited127-case six-order A/A/B/C campaign, all1067 precision inputs, and67 accepted actualTBE cases, no selection or failed-control rerun. A/C official; B candidate. General body unchanged. No timing inference from code size.
