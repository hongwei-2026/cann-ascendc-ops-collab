#!/bin/bash
set -euo pipefail
R=/mnt/workspace/2329-JadeYara-20261104-1812/J079
P=/mnt/workspace/2329-JadeYara-20261104-1812/J078
trap 'rc=$?; printf "%s\n" "$rc" > "$R/phase5.exit"' EXIT
source /home/developer/Ascend/cann/set_env.sh
OPP="$P/build-abc/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/j078_abc_custom"
export ASCEND_CUSTOM_OPP_PATH="$OPP"
export LD_LIBRARY_PATH="$OPP/op_api/lib:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:$ASCEND_HOME_PATH/lib64"
msprof --output="$R/taskonly-swapped/profile" --ai-core=off --task-time=on "$P/control-abc/device_probe" "$R/swapped-plan.txt" 0 > "$R/logs/perf-taskonly-swapped.log" 2>&1
python3 "$R/analyze_task_only.py" "$R/swapped-manifest.json" "$R/logs/perf-taskonly-swapped.log" "$R/taskonly-swapped/profile" "$R/taskonly-swapped-summary.json"
msprof --output="$R/taskonly-return/profile" --ai-core=off --task-time=on "$P/control-abc/device_probe" "$R/normal-plan.txt" 0 > "$R/logs/perf-taskonly-return.log" 2>&1
python3 "$R/analyze_task_only.py" "$R/diagnostic-manifest.json" "$R/logs/perf-taskonly-return.log" "$R/taskonly-return/profile" "$R/taskonly-return-summary.json"
