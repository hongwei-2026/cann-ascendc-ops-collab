# J008 归档传输记录

设备作业已于20:12完成，退出码0。完整归档26,041,425字节，云端SHA256为e460456f32edd4bdada3a0f671d6d14d77c61eb376c681fa85a7f331f1ae20f3。

首次SSH分段下载已保存前13个1MiB分段；读取offset=13631488时返回码255、仅收到671744字节，stderr报`Bad packet length 2076273430`和`Connection corrupted`。失败分段未写入已完成集合。随后仅续读未完成分段，没有重启编译/设备测试、没有改变网络设置。最终是否完整恢复以最终SHA256及ZIP CRC校验为准。

第二次续传成功到18MiB，在offset=18874368处再次发生Bad packet length/Connection corrupted，仅收到344064字节。后续剩余分段改为远端每16KiB输出后间隔20ms，以减小单次传输突发；这是本次只读传输的节奏调整，不改变SSH/代理/网络配置。
