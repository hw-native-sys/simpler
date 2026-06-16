# Scheduler-Overhead Model

The timeline model behind `sched_overhead_analysis` (text report) and the
`swimlane_converter --overhead` track (8 Perfetto counter lines). Both answer
one question: **when is makespan time wasted because a free core has ready work
the scheduler hasn't placed — vs. legitimately busy or dependency-limited?**

Both tools need two artifacts, captured in **separate** runs (co-running them
perturbs the swimlane timing):

1. `l2_swimlane_records.json` — per-task timing, from `--enable-l2-swimlane`
   (level ≥ 3 for the scheduler-loop parts).
2. `deps.json` — the task DAG, from a separate `--enable-dep-gen` run.

## Timeline basis

- **Window (makespan)** = `[first aicore start, last aicore end]` =
  `[min(start), max(end)]`. AICore start/end timestamps are authoritative.
- Per task: `dispatch` (AICPU wrote the descriptor) → `start` (core began the
  kernel) → `end` (kernel done) → `finish` (AICPU observed completion).
- All shares are **% of the makespan** (dependency-aware), never core-time —
  core-time treats an idle engine as waste even when its work is genuinely done.

## Readiness (`ready`)

`ready(T) = max over T's producers of producer end_time` — the data exists once
the producer kernel ends. Two refinements that matter:

- **Off-perf predecessors.** A task whose predecessors are all absent from the
  perf set (e.g. host/DMA input loads) falls back to **its own dispatch**, NOT
  the window start. Defaulting to `w0` would mark it "ready from t=0" and invent
  ~2% of false early overhead during the dispatch ramp.
- **MIX tasks.** A task with records on **both** engines (one `task_id`, AIC +
  AIV) counts as ready work for **both** — it needs both engines to launch.
  Attributing it to one engine (its first record) hides the other engine's wait.

## Overhead (the core metric)

Per instant, for each core type `T` (AIC / AIV) independently:

```text
overhead(T)  ⇔  idle T-core exists (k_T − running_T > 0)
                AND a ready, UNDISPATCHED T-task exists ([ready, dispatch])
```

- A dispatched-but-not-started task is **not** counted — the scheduler already
  placed it; its `[dispatch, start]` pickup is *aicore switch* (below), not
  overhead.
- An engine with **no ready work** (`ready == 0`) is **not** overhead — its idle
  cores are dependency-mandated (e.g. AIV idle through an AIC-heavy tail). That
  cost shows as low parallelism, not as wasted scheduler time.

System aggregates over the present engines:

| Line | Definition |
| ---- | ---------- |
| `all_overhead` | **every** present engine is overhead (whole chip blocked — e.g. a MIX waiting to launch) |
| `has_overhead` | every engine that **has ready work** is overhead (engines with no work ignored) |

`all_overhead ≤ per-engine ≤ has_overhead`. On qwen3-14b decode_layer
(a2a3, 542 tasks): AIC 15.0%, AIV 10.3%, `all` 3.3%, `has` 20.2%.

## aicore switch

On a core, the gap `[prev_end, start]` of a task whose `dispatch < prev_end`
(pre-dispatched / pending pickup). The core picks up an already-issued task; the
gap is the pickup latency, ~0.8 µs each.

- Report it **per core** (~8–11 µs/core), **never the all-cores sum** (~240 µs
  reads as a scary aggregate but switches on different cores overlap).
- A switch is **overhead** when the engine has other ready work at that instant
  (the idle pickup core coincides with undispatched-ready work), else
  **independent**. On the sample data ~63% of AIC switch falls in overhead.
- **Makespan switch bound**: `lower = min over all cores`,
  `upper = sum of per-engine minima` (best core per phase on the critical path).
  Sample: `[0, 8.10 µs]` = `[0%, 0.8%]` of makespan — the switch the makespan
  truly pays is tiny; most switch time overlaps overhead anyway.

## Tooling

### `swimlane_converter --overhead`

Adds 8 counter tracks under the **AICPU Scheduler** process (`pid=3`,
`oh_`-prefixed) so they overlay the AICore task bars:

```text
oh_{aic,aiv}_idle      core count not executing (k − running)
oh_{aic,aiv}_ready     undispatched-ready task count (MIX counts for both)
oh_{aic,aiv}_overhead  0/1 = idle>0 AND ready>0
oh_all_overhead        0/1 = every engine overhead
oh_has_overhead        0/1 = every working engine overhead
```

```bash
python -m simpler_setup.tools.swimlane_converter <perf>.json \
    --deps-json <deps>.json --overhead -o out.json   # drag into ui.perfetto.dev
```

### `sched_overhead_analysis`

| Part | Content |
| ---- | ------- |
| 1 | Overhead verdict — per-engine + system `all`/`has` overhead (% of makespan) |
| 2 | aicore switch — per-core min/mean/max, overhead-vs-independent split, makespan bound |
| 3 / 4 | Head OH / Tail OH distributions |
| 5 | AICPU scheduler-loop budget — ns/loop, phase split, pop hit-rate, fanout/fanin |
| 6 | Critical-path attribution — compute vs scheduler-injected µs on the makespan path |

```bash
python -m simpler_setup.tools.sched_overhead_analysis \
    --l2-swimlane-records-json <perf>.json --deps-json <deps>.json
```
