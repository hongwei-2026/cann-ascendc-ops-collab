# J069: continuous Gather weights, same actual-TBE mathematics
PReluGradUpdate_2329 / JadeYara / performance stage / 2026-11-07.
Goal: one isolated structural candidate on precision-correctJ068: keep global32B contiguous block partition, replace repeated short channel tiles with batch weight window + exactinteger Gather. No change to the actualTBE arithmetic DAG.
Only J069 and personal ledger/dev-log/checkpoint may change. ProductionJ035 and all prior experiments frozen. User retains official submission. No commit/push.
A/C frozenJ068; B adds a bounded ND same-input/one-dimensional channelweights route for many-short-tile cases. Shared weights, efficient alignedS1 resident/splitC and one/fewsegment work stay on the previous paths. New metadata/keys independent of old tiling ABI.
Use same builtinTBE binary oracles plus new phase/oddC/middleS cases, then one sameprocess sixorder completeTask. Preserve all control failures. Format/broadcast fullNPU fallback remains separately designed, not advertised implemented here.
User authorized this project's source and test evidence transfers between the same verified910C and local machine. Fixedhost verification remainsmandatory. Current officialforward is being restored; local preparation only until it is usable.
