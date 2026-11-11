# C008 — CodeArts TBE 与 API 能力盘点

本闸只读检查 CodeArts 的 910B / CANN 9.1 环境，不构建候选、不修改 Kernel，也不把平台 leaderboard 的 best time 当作 TBE 基线。

## 运行

先把本机刚推送的 `main` 拉到 CodeArts 协作仓，再运行：

```bash
cd /mnt/workspace/cann-ascendc-ops-collab
GIT_EDITOR=true git pull --ff-only origin main
source /home/developer/Ascend/cann-9.1.0/set_env.sh
bash people/hongwei-2026/ConfusionTransposeD_2323/workdir/code/tests/run_c008_preflight.sh
```

CodeArts本次确认的CANN环境路径是`/home/developer/Ascend/cann-9.1.0`。脚本会检查芯片和CANN版本、TBE文件/注册符号、`Transpose`/`Gather`头文件声明；不访问平台私有数据，也不猜隐藏点shape。

## TBE入口续查

初次盘点找到TBE源码后，在同一终端继续读取实现注册和可调用入口：

```bash
CTD_CANN="${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.1.0}"
CTD_TBE="$CTD_CANN/opp/built-in/op_impl/ai_core/tbe/impl"
for f in \
  "$CTD_TBE/ops_legacy/confusion_transpose_d.py" \
  "$CTD_TBE/ops_math/dynamic/confusion_transpose_d_apt.py" \
  "$CTD_TBE/ops_math/ascendc/confusion_transpose_d/confusion_transpose_d_apt.cpp"; do
  echo "===== $f ====="
  grep -nE -C 2 'register_operator|def |kernel_name|op_type|transpose_first|transposeFirst|ConfusionTransposeD|__aicore__|__global__|DataCopy|Transpose|Gather|Tiling|Launch' "$f" | head -n 160
done
echo '===== Public API files and references ====='
find "$CTD_CANN/include" -type f -iname '*confusion*transpose*' -print
grep -RInE 'aclnnConfusionTransposeD|aclnnConfusionTranspose|npu_confusion_transpose' \
  "$CTD_CANN/include/aclnnop" "$CTD_TBE" 2>/dev/null | head -n 120 || true
python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("torch_npu")
print("torch_npu_spec:", spec)
if spec is not None:
    try:
        import torch_npu
        print("torch_npu_version:", getattr(torch_npu, "__version__", "unknown"))
        print("matching_symbols:", [name for name in dir(torch_npu) if "confusion" in name.lower() or "transpose" in name.lower()])
    except Exception as exc:
        print("torch_npu_import_error:", repr(exc))
PY
```

把这段命令的完整输出贴回，用来判断TBE能否单独执行以及是否有公开API；源码匹配结果本身不算性能基线。

## 回传

从终端复制完整的 `C008_EVIDENCE_BEGIN` 到 `C008_EVIDENCE_END`。完整日志包会保存到 `/mnt/workspace/exp_logs/ctd2323/`。若可以下载，把 `EXP_C008_*.tar.gz` 放入 `tasks/active/ConfusionTransposeD_2323/explorations/logs/EXP_C008/` 后提交；若仍不能下载，回传 marker 原文并标成粘贴证据。

## 结案条件

- 找到可在同一 910B / CANN 9.1 设备调用的 TBE 实现，并可获得与 P004 相同输入上的 kernel task 时间：进入下一条有界 Kernel 假设前，补齐 P004/TBE 对照。
- 找不到 TBE 实现、不能绕开自定义 op 注册冲突，或缺少可映射的目标输入：在 60 分钟预算内止损并写 DEC；不改 Kernel，不声称平台最优就是 TBE。
