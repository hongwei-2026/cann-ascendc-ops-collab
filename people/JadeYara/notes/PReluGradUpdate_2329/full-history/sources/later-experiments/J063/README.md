# J063: measured, not promoted

See DEC-J063.md for final cloud results. Precision passed, all enabled ShortS rounds won, but BF16 S1 fallback regressions remain and strict performance controls failed13/270. Production J035 remains untouched.

# J063: ShortS independent template keys (not device-validated)

Prepared 2026-11-07. This is a new isolated candidate, not a replacement for J035 and not ready for official submission.

- Standard operator name PReluGradUpdate; 43/46 original files remain byte-identical to production J035. Changes: Host route/budget, key declaration, kernel entry. One new header short_s.h is byte-identical to measured J062.
- Host keys20/21/22 cover per-channel weights, 2<=S<=32, blockFactor>S. J062 arithmetic, local-memory layout, tile geometry, core count and conversion rounding are unchanged. S=1, shared weights and S>32 retain J035 dispatch.
- Rationale: preserve the existing multi-process short-S gain while separating its instantiation from the original default fallback. New keys can still alter object layout; no claim that this cures the earlier per-process timing anomaly.
- DeepSeek E0 established non-reproduction of two BF16 fallback anomalies in two reruns. It does not establish a unique runtime/placement cause. Original control failures are retained. Hostkey evidence remains insufficient for cloud access.
- Local checks and raw results will be saved here. CANN compilation, NPU precision, pressure, performance, compiled fallback correspondence and official results are NOT RUN.
- No SSH/SCP attempt, no host-check relaxation, no source merge, no commit/push or official submission.

Offline checks completed:5020 geometry cases,447 weight-mapping tiles/415604 elements,18 phase-reuse checks; exit0. Full Chinese review:review-20261107.md. Device tests remain blocked and NOT RUN. candidate/build.sh is an unexecuted private build entry, not a validation result.

## Cloud continuation, 2026-11-07 20:12 CST
User supplied the target WebIDE ED25519 fingerprint SHA256:qPhIpfPEmRfFhfArnWWBP/aRrj0JXj7vr70EAlwK4i0. The live SSH handshake matched exactly. StrictHostKeyChecking stayed yes, UpdateHostKeys no; a dedicated known_hosts entry was created only after exact fingerprint verification. Cloud access is restored through the same official AtomGit forwarding path. Windows keyscan waited for a server banner while this forwarding path required the client banner first; this was not evidence that the cloud instance was offline.

Uploaded device-source.zip SHA256 407911c1fab0c73005054fe963206cc7b8cb0b7099c59ef9f4c9d3c461a19a1a (180720 bytes) to the existing private experiment root, then created a new J063 directory. Build PID 6738. ABC A/C source remains byte-identical to J062 frozen J035; B mechanically aliases candidate. Also build the real standard operator and test the same golden once per input through its real API. Existing 924-case precision and 45-case performance inputs are reused. Only historical split-C metadata is corrected to actual Host geometry. Device build and tests are IN PROGRESS, not a pass claim.