# J069 measured: continuous Gather, actual-TBE mathematics

Original910C reconnected successfully using the existing strict verified host pin. No official submission, production replacement, commit or push.

## Build and precision
Standard/ABC all,binary,package passed. Standard/B54/54 and A/C45/45 function correspondences passed. Existing15 live B functions differ fromA at the byte level but retain equal sizes; no unique latency cause is inferred.
Two corrected precision plans contain1063groups each:970 correctedlegacy,27actual builtinTBEoracle groups,57finitephase/window inputs,9newactualTBEspecial groups. ABC4584calls and standard1528calls all passed. Actual special goldens were generated with ASCEND_CUSTOM_OPP_PATH removed.

## Structural performance
Baseline is precision-correctJ068, NOT J035 or an officialscore.127inputs,3048windows,1066800NPUcalls completed. No control-failure rounds were dropped.
All53Gather inputs improved in all318rounds by >0.1us AND >2percent. Median per-input speedup86.135percent; range38.593..99.202percent; median saved74.124375us. FP32/FP16/BF16 medians88.379/84.666/87.223percent.
All74untouched inputs'444rounds passed all-four control bounds; median gain0percent, worst -0.029875us. Overall strictcontrols753/762, A/A757/762. Nine unstable baseline-control rounds are retained in rawresults.
Examples A->B:
- FP32[8,64,3]:11.48025->6.20025us; FP16:11.590125->5.5us; BF16:13.167625->5.5005us.
- FP32[16,63,33]:17.42->6.76075us.
- FP32[127,1024,3]:1320.21875->12.161us; BF16:1651.488->13.1805us.
- Untouched FP32[4097,128,1]:12.38025->12.38025us.
All detailed cases/rounds, controls and interpretation method are in performance-interpretation.json.

## Concrete defects repaired and evidence retained
1. int32 ShiftRight requires an int32 scalar shift, not uint32. Original smoke failure retained; correctedsmoke binary passed.
2. CANN9.3 REGISTER_TILING_FOR_TILINGKEY parser requires a quoted TILING_KEY_VAR expression, not TILING_KEY_LIST. kernel_info_infer.py:131-147,210-220 establishes this.
3. UI_LIST encodes listordinals, notschMode values (template_argument.h:328-351). GeneralGather schMode24/25/26 is encoded15/16/17 with all3dtypes,10/11 withFP16only,5 withFP32only. Correct conditional registrations compiled and were the source for finalprecision/performance. Earlierprecisionpassed too, but its incompleteperformance is explicitly aborted and not interpreted.
4. /home196GBfilled during the abortedrun. Its710files/721535724B were moved intact to the existing/mnt/workspace volume, eachfileSHAverified. No data discarded or storage purchased. Newprofiling also uses that existingvolume. Previousprecision logs remain in *-pre-metadata-fix directories.

## Artifacts and remaining scope
Cloud complete evidence archive37,424,049B/581files SHA2566210da2d5f8fdce1d45e4f1ab719dfdcca2c1d227728242bac9995a64a137b83 has completed2MiBchunk/totalSHA/CRC verification and is available locally as device-evidence.zip. The standard48file code/root package is candidate-standard-source.zip (69284B,SHA2567c7bf2954376f0ae7bb34cebe1e4faa27f9291004d750bacb40562ff26bb60ae). All48sourcefiles exactly match the finalcloud evidence.
This candidate still implements the priorND/same-g-x/1Dchannelweight engine, plus its new boundedC/mediumS route. It preserves actualTBE arithmetic and does not claim fullformat/broadcast functionality. UnchangedJ068 shared/longS costs relative toJ035 are not fixed by this experiment. Do not turn these private numbers into an officialscore claim or infer hidden officialshapes.
J070 now implements the CPU-proved generalbroadcast fallback in a separate directory, preserving allJ069 fastpaths. ProductionJ035 remains frozen.
