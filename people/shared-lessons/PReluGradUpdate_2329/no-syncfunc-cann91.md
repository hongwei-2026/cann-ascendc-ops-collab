# CANN 9.1：不要用 SyncFunc

`SyncFunc<HardEvent::S_V>()` → `error: use of undeclared identifier 'SyncFunc'`（opc 直接挂）。

标量读 GM/`GetValue` 之后、向量 `Duplicate`/`Muls` 之前：

```cpp
event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
SetFlag<HardEvent::S_V>(e);
WaitFlag<HardEvent::S_V>(e);
```

见 EXP006 / EXP_P017 日志。
