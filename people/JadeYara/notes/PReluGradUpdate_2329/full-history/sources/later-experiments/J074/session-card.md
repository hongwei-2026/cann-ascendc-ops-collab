# J074: investigate J073 single control anomaly
2026-11-08. JadeYara sole implementer; performance investigation,90min bounded.
User: investigate the J073 C-side0.14us anomaly and assess whether J073 may be trial-submitted. Preserve all J073 records, source and production; no official submission/commit/push.
Scope: J074 investigation notes/scripts/data, personal ledger/dev-log. No candidate source change. Start by raw original CSV/timestamps/order/metrics, binary correspondence, old profiler frequency and telemetry availability. Then preregister one falsifiable diagnostic before device execution; no selective retries or discarded failures.
Initial facts: case105 FP16[64,96,33],round5 last C=12.48 vs A1/A2=12.61/12.62,B=12.62. All round6 aliases=12.48..12.49. AIV/vector times did not show the same0.14us shift in aggregate medians. This is temporal evidence, not yet proof of clocks,temperature,cache or scheduling as root cause.
