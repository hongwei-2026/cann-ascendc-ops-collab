# 云端日志下载位置

`run_c006_cloud.sh`完成或失败时，将完整`EXP_C006_*.tar.gz`复制到此目录，便于从CodeArts项目文件树下载；同时保留原explorations与exp_logs目录内原包。此目录中的运行原包默认不纳入Git，取得后应按本题约定归档到tasks日志目录。

如果下载仍不可用，终端会自动打印`C006_EVIDENCE_BEGIN`至`C006_EVIDENCE_END`；先粘贴该段即可查看原版/候选功能、设备profile和错误摘要。摘要并非完整原始云日志，不得宣称完整归档。
