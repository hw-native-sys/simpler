# Chasing down a capacity code (1, 2, 3, 4, 11, 12)

← [Device Error Codes](../device-error-codes.md)

SCOPE_DEADLOCK, HEAP_RING_DEADLOCK, FLOW_CONTROL_DEADLOCK,
FANIN_CAPACITY_EXCEEDED, TENSORMAP_OVERFLOW and SUBMIT_CAPACITY_EXCEEDED all mean that a
runtime resource could not admit more graph state. The adjacent device-log line identifies
the actual resource and determines how strong that diagnosis is:

- `Provable head-of-line deadlock` is a structural proof: in TRB the reclaim head
  is the oldest task owned by an open scope on that ring, and the blocked
  orchestrator cannot end the scope that pins it. On A5, this verdict is reached
  only after at least 10 ms without reclaim progress and an exact-watermark
  publication acknowledgment.
- `No reclaim progress for ~500 ms` or `cannot reclaim space after ~500 ms` is
  the backstop. It proves that reclaim remained stalled, but not whether the root
  cause is undersizing, a stuck consumer, or a stalled scheduler.
- `Graph Too Large` / `Fanin Capacity Exhausted` / `TensorMap Entry Pool
  Exhausted` is HBG, and it is unambiguous: that runtime builds a
  whole-graph-resident image on the host, so the graph simply does not fit. These
  checks return immediately; there is no concurrent scheduler progress that could
  free capacity while the host is building. `Graph Heap Exhausted` shares the
  wording but not the meaning — HBG's heap allocator is handed the whole virtual
  window during a real bind, so it does not report exhaustion there; the graph
  heap's size shows up instead as a failed device-region commit after
  orchestration.

Do not guess at the ring sizes from the error code alone. Turn on `scope_stats`,
which records the high-water mark of all four resources (task-window slots, heap
bytes, dep-pool entries, tensormap entries) per `SIMPLER_SCOPE`:

```python
cfg = CallConfig()
cfg.enable_scope_stats = True
worker = Worker(..., output_root="outputs")
worker.run(callable, args, cfg)
```

It works on a **failing** run: the metadata line is marked `"fatal": true` and
everything written before the fatal is kept. So point it straight at the workload
that trips the code.

| Runtime | Bottleneck resource | Code | Fix |
| ------- | ------------------- | ---- | --- |
| HBG | task count | 3 | raise `runtime_env.ring_task_window` (any positive count), or shrink the graph |
| HBG | inline fanin | 4 | reduce distinct producers to `CHIP_MAX_FANIN` (currently 128) or less; HBG has no dependency spill pool |
| HBG | TensorMap entries | 11 | increase `CHIP_TENSORMAP_POOL_SIZE`, or reduce registered outputs |
| HBG | submit-time capacity | 12 | see the four sources below |
| TRB | open-scope task window | 1 or 3 | raise `runtime_env.ring_task_window` (a power of two, >= 4), split the scope, or diagnose stalled reclaim |
| TRB | heap | 2 | raise `runtime_env.ring_heap`, shrink allocations, or diagnose stalled reclaim |
| TRB | dependency pool | 4 | raise `runtime_env.ring_dep_pool`, cut fanin, or diagnose stalled reclaim |

HBG's graph heap is absent from the table because it has no knob and cannot latch
code 2: orchestration allocates it out of a virtual window, and the device region
is committed afterwards at the measured size. A graph too large for the device
fails at that commit, on the host, naming the byte count it asked for.

## SUBMIT_CAPACITY_EXCEEDED (12) sources

Code 12 is latched on the host while a submission is being admitted, so it has no
device-log line at all. The `LOG_ERROR` immediately above it names which of four
capacities ran out:

- **Task slots or heap bytes.** The allocator has no free slot, or the heap cannot
  cover what this submission owns. Raise `runtime_env.ring_task_window`, or shrink the
  sub-task outputs.
- **The compact argument pools.** A boundary draws more tensor or scalar pool entries
  than the single task slot it occupies is worth. Reduce its parameter count.
- **The owned heap request.** The bytes one Definition owns exceed `INT32_MAX`. Split
  the body.
- **The Definition cache.** The per-worker `GRAPH_MAX_DEFINITIONS` limit was reached.
  Submit fewer distinct Definitions, or raise the constant.

The runtime's own `error hint:` line already names the knob for the code it
latched; this table is for when you have `scope_stats` output and want to read the
peak back to a knob.

Report fields, the `Top Peaks` table and the plotting tool are documented in
[`../../dfx/scope-stats.md`](../../dfx/scope-stats.md).
