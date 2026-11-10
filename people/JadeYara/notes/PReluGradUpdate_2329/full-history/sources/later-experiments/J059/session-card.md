# J059: fixed-4096 FP32 three-stage pipeline
PReluGradUpdate_2329 / JadeYara. One authorized experiment, 2026-11-07 CST.
Goal: compare complete Task for J035 vs two FP32 input slots and two output slots, real next-batch MTE2 issue before current Vector.
Scope: independent local/remote J059 directories and task/personal ledger plus dev-log. Baseline workdir stays frozen.
Keep Host, tile4096, 40-core target, resident weights, Compare/Select mathematics and every PIPE_V unchanged. Never scan parameters.
Two 16384B input strides retain both slot phases. Move grads' 256B padding after the whole queue. UB=185088B, available188416B.
A=J035, B=full pipeline, C=independent J035 alias; same-process six balanced orders with A/A control.
Run necessary correctness and a single performance campaign; inspect timeline capability and actual dependency evidence. Stop if tiny or slower.
No official submission, commit, push, or modification of other experiments. User decides promotion. Stop That Shit 0.2.5 read; no claim Guard is armed.
