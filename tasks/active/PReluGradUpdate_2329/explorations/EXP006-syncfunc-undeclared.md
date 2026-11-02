# EXP006 — CANN 9.1 无 `SyncFunc`

| 字段 | 内容 |
|------|------|
| 题目 | PReluGradUpdate_2329 |
| 日期 | 2026-11-02 |
| 日志 | `explorations/logs/EXP_P017_20261102_172942/` |
| 结论 | Claude 代码里的 `SyncFunc<HardEvent::S_V>()` **无法编译**；改为 `SetFlag`/`WaitFlag` |

## 现象

```text
p_relu_grad_update.h:154:9: error: use of undeclared identifier 'SyncFunc'
p_relu_grad_update.h:226:9: error: use of undeclared identifier 'SyncFunc'
Kernel Compilation Error: OpType PReluGradUpdate
```

三条 tiling key（fp32/fp16/bf16）全部 opc 失败，未进入 `-e` 样例。

## 处置

```cpp
event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
SetFlag<HardEvent::S_V>(evtSV);
WaitFlag<HardEvent::S_V>(evtSV);
```

用在 `GetValue`/`ReadWeight` 标量读之后、`Duplicate`/`Muls` 向量写之前。
