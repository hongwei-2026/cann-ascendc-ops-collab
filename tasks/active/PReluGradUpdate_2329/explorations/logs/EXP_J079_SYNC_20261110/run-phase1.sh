#!/bin/bash
set -euo pipefail
R=/mnt/workspace/2329-JadeYara-20261104-1812/J079
P=/mnt/workspace/2329-JadeYara-20261104-1812/J078
trap 'rc=$?; printf "%s\n" "$rc" > "$R/phase1.exit"' EXIT
source /home/developer/Ascend/cann/set_env.sh
OPP="$P/build-abc/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/j078_abc_custom"
export ASCEND_CUSTOM_OPP_PATH="$OPP"
export LD_LIBRARY_PATH="$OPP/op_api/lib:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:$ASCEND_HOME_PATH/lib64"
python3 "$R/prepare_diagnostic.py"
ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 LD_DEBUG=libs "$P/control-abc/device_probe" "$R/trace-plan.txt" 0 > "$R/logs/trace-original.log" 2>&1
msprof --output="$R/original/profile" --aic-metrics=PipeUtilization --task-time=on "$P/control-abc/device_probe" "$R/normal-plan.txt" 0 > "$R/logs/perf-original.log" 2>&1
python3 "$P/abc/control/campaign.py" analyze "$R/diagnostic-manifest.json" "$R/logs/perf-original.log" "$R/original/profile" "$R/original-summary.json"
