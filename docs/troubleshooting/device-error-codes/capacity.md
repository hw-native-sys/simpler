# Chasing down a capacity code (1, 2, 3, 4, 11)

← [Device Error Codes](../device-error-codes.md)

SCOPE_DEADLOCK, HEAP_RING_DEADLOCK, FLOW_CONTROL_DEADLOCK,
DEP_POOL_OVERFLOW and TENSORMAP_OVERFLOW all mean that a runtime resource could
not admit more graph state. The adjacent device-log line identifies the actual
resource and determines how strong that diagnosis is:

- `Provable head-of-line deadlock` is a structural proof: in TRB the reclaim head
  is the oldest task owned by an open scope on that ring, and the blocked
  orchestrator cannot end the scope that pins it. On A5, this verdict is reached
  only after at least 10 ms without reclaim progress and an exact-watermark
  publication acknowledgment.
- `No reclaim progress for ~500 ms` or `cannot reclaim space after ~500 ms` is
  the backstop. It proves that reclaim remained stalled, but not whether the root
  cause is undersizing, a stuck consumer, or a stalled scheduler.
- `Task Capacity Exhausted` / `Graph Heap Exhausted` / `Fanin Capacity Exhausted`
  / `TensorMap Entry Pool Exhausted` is HBG, and it is unambiguous: that runtime
  builds a whole-graph-resident image on the host, so the graph simply does not
  fit. These checks return immediately; there is no concurrent scheduler
  progress that could free capacity while the host is building.

Do not guess at the ring sizes from the error code alone. Turn on `scope_stats`,
which records the high-water mark of all four resources (task-window slots, heap
bytes, dep-pool entries, tensormap entries) per `PTO2_SCOPE`:

```python
cfg = CallConfig()
cfg.enable_scope_stats = True
cfg.output_prefix = "outputs/my_run"
worker.run(callable, args, cfg)
```

It works on a **failing** run: the metadata line is marked `"fatal": true` and
everything written before the fatal is kept. So point it straight at the workload
that trips the code.

| Runtime | Bottleneck resource | Code | Fix |
| ------- | ------------------- | ---- | --- |
| HBG | task capacity | 3 | raise `ring_task_window` (`PTO2_RING_TASK_WINDOW`), or shrink the graph |
| HBG | graph heap | 2 | raise `ring_heap` (`PTO2_RING_HEAP`), or shrink intermediate tensors |
| HBG | inline fanin | 4 | reduce distinct producers to `PTO2_MAX_FANIN` (currently 128) or less; HBG has no `PTO2_RING_DEP_POOL` |
| HBG | TensorMap entries | 11 | increase `PTO2_TENSORMAP_POOL_SIZE`, or reduce registered outputs |
| TRB | open-scope task window | 1 or 3 | raise `PTO2_RING_TASK_WINDOW`, split the scope, or diagnose stalled reclaim |
| TRB | heap | 2 | raise `PTO2_RING_HEAP`, shrink allocations, or diagnose stalled reclaim |
| TRB | dependency pool | 4 | raise `PTO2_RING_DEP_POOL`, cut fanin, or diagnose stalled reclaim |

The runtime's own `error hint:` line already names the knob for the code it
latched; this table is for when you have `scope_stats` output and want to read the
peak back to a knob.

Report fields, the `Top Peaks` table and the plotting tool are documented in
[`../../dfx/scope-stats.md`](../../dfx/scope-stats.md).
