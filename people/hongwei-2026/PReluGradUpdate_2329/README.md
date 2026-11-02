# hongwei · PReluGradUpdate_2329

- **云端同步用这一版**：`workdir/code/`  
- 共享台账 / 材料 / EXP：`tasks/active/PReluGradUpdate_2329/`（只读参考 + 可往共享台账补 EXP）  
- 共识教训：`people/shared-lessons/PReluGradUpdate_2329/`  

当前实现已使用独立 neg/fg buffer，并将 CANN 9.1 不支持的 `SyncFunc` 替换为 `SetFlag`/`WaitFlag`。

CodeArts 每次覆盖工程后必须运行：

```bash
bash people/hongwei-2026/PReluGradUpdate_2329/verify_codearts_source.sh \
  /mnt/workspace/cann-ascendc-ops-collab /mnt/workspace/prelu2329/code
```

只有输出 `CodeArts source verification: PASS` 才允许开始构建。
