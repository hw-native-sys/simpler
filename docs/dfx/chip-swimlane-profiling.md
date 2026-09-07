# Chip Swimlane Profiling — Per-task Timing & Scheduler Phases

> **Lighter alternative for a single interval.** If you only need the
> dispatch→finish window of one or two specific tasks (not a full per-task
> timeline), prefer the selective **task-timing slots**: tag the task with
> `CoreTaskArgs::set_task_timing_slot(0..15)` and read the
> `…device_wall.task_slot_<N>` `[STRACE]` span. It reuses the fixed device-phase
> buffer — no collector threads, no per-task AICore records, works in
> `SIMPLER_DFX=0`, and avoids the ~0.8 µs/switch observer effect below. See
> [device-phases.md](device-phases.md#selective-task-timing-slots-implemented)
> and [l2-timing.md](l2-timing.md#4b-measuring-one-tasks-dispatchfinish-without-the-swimlane).
> Use the full swimlane (this doc) when you need **every** task's start/end and
> the dependency/scheduler-phase picture.

## 1. Background & Motivation

Why a kernel takes the time it takes is rarely visible from
end-to-end runtime numbers. Two cases dominate the profiler diet:

- **Task-level timing.** When kernel A is slow, is the kernel slow,
  or is something else holding it up? Where does each AICore task
  start and end on the wall clock, and what fanout / fan-in chain
  does it sit in?
- **Scheduler overhead.** Inside AICPU's scheduling loop, time is
  split across "process completed tasks", "dispatch ready tasks",
  "incremental scan for roots", and "idle wait". When the AICPU
  thread is hot, knowing which of those four phases dominates
  pinpoints the fix.

chip swimlane profiling captures both: per-task `(start, end,
dispatch, finish)` records, plus per-iteration phase records from the active
AICPU or AICore scheduler producer and per-submit orchestrator
envelopes. The host writes a Chrome Trace Event JSON
that loads directly in Perfetto. For the scheduler-overhead deep-dive, capture
`deps.json` separately and invoke `sched_overhead_analysis` explicitly; see its
[tool documentation](../../simpler_setup/tools/README.md#sched_overhead_analysis).

## 2. Overview

- **Per-task AICore timing** — `start_time_us`, `end_time_us`,
  `duration_us`, plus AICPU-stamped `dispatch_time_us` / `finish_time_us`.
- **Per-task dependency arrows** — successor edges are NOT recorded
  in the swimlane record itself (the device hot path stays clean —
  see PR #863). Instead, `swimlane_converter` joins
  `chip_swimlane_records.json` with `deps.json` from
  [`dep_gen`](dep-gen.md) at post-process time; see
  [§3.5](#35-dependency-arrows-from-dep_gen).
- **Scheduler phases** — producer-specific per-iteration breakdown. AICPU uses mutually
  time-exclusive **outer** phases (`complete` / `async_poll` / `dispatch` /
  `release` / `dummy` / `early_dispatch` / `drain` / `graph_prepare`), plus
  nested phases.
  In `tensormap_and_ringbuffer`, `resolve` is nested within `complete` or
  `dummy`; in `host_build_graph`, `resolve_standalone`, `async_poll`, and
  `dummy` are standalone, mutually exclusive phases on the dedicated P
  thread. The converter renders `resolve_standalone` as `resolve` on that P
  thread's main scheduler lane; TMR's nested `resolve` uses a sibling
  scheduler sub-lane. The drain sub-phases are nested within
  their `drain` bar,
  and two **separate-lane**
  phases (`dummy_task` and `predicated_skip`, sampled immediately before
  `on_task_complete()` begins dependency resolution and rendered as synthetic
  0.02 us markers on Worker View AICPU_N rather than on the sched lane).
  `predicated_skip` identifies a real task whose dispatch predicate evaluated
  false. Its marker uses the task's ordinary function name and carries
  `predicated_pass: false` in its Perfetto arguments; a predicate that evaluates
  true follows the ordinary task timing path with no special argument. The
  source `predicated_skip` phase remains in `chip_swimlane_records.json` and is
  not copied into the merged Worker View event's arguments.
  `dummy_task` is emitted by both a2a3 runtimes and by a5
  `tensormap_and_ringbuffer`; `predicated_skip` is emitted by the a2a3 and a5
  `tensormap_and_ringbuffer` runtimes, where predicated dispatch is
  implemented. a5 `host_build_graph` has no dummy-task phase path. Idle
  iterations no longer emit a record on a2a3; the host tooling reconstructs
  idle spans from the gap between consecutive work records on the same thread.
  See §3.2 for the full per-phase table. Legacy captures may carry `scan` /
  `poll` / `idle` / `fanout` / `prestage` — current a2a3 builds no longer
  emit them (PR #1079's Scan/Poll debug overlay was removed;
  Fanout was renamed Resolve and now also filters out <1 µs walks;
  Prestage was renamed EarlyDispatch).
- **Orchestrator submit envelope** — one record per `submit_task()`
  / `alloc_tensors()` call covering the whole submit's
  `[start, end]` window (`orch_submit` phase). Per-sub-step
  cumulative cycles (sync / alloc / params / lookup / insert /
  fanin) still live in the cold-path device log via the
  `g_orch_*_cycle` counters — that's where you go for "which
  sub-step dominates overall"; the per-submit record covers
  "which submit was slow".
- **Standard outputs** — raw `chip_swimlane_records.json`, plus a
  Perfetto-loadable `merged_swimlane_*.json` produced by
  `swimlane_converter`.

Enable in one line:

```bash
python tests/st/<case>/test_<name>.py -p <platform> -d 0 --enable-chip-swimlane
```

## 3. How to Use

### 3.1 Enable chip swimlane

`--enable-chip-swimlane` accepts an optional integer **perf_level**
(0–4). A bare flag defaults to level 4 (full collection,
backward-compatible with the old boolean behavior).

| Level | Collects | Notes |
| ----- | -------- | ----- |
| 0 | Nothing (disabled) | Default when flag is absent |
| 1 | AICore timing only (start_time_us/end_time_us/task_id/func_id/core_type) | No Scheduler timestamps |
| 2 | + Scheduler per-task dispatch_time_us, finish_time_us | Producer is identified as `aicpu` or `aicore` |
| 3 | + scheduler phases (`scheduler_records`) | Skips orchestrator phases |
| 4 | + orchestrator phases (`aicpu_orchestrator_phases[]`) | Full collection |

Dependency arrows are not produced by any swimlane level — see
[§3.5](#35-dependency-arrows-from-dep_gen) for the dep_gen join.

```bash
# Standalone runner
python tests/st/<case>/test_<name>.py -p <platform> -d 0 --enable-chip-swimlane [PERF_LEVEL]

# pytest — same flag shape
pytest tests/st/<case> --platform <platform> -d 0 --enable-chip-swimlane [PERF_LEVEL]

# Bare flag (no integer) — shorthand for level 4 (full collection)
python tests/st/<case>/test_<name>.py -p <platform> -d 0 --enable-chip-swimlane
```

- `<platform>` — one of `a2a3` / `a2a3sim` / `a5` / `a5sim`; the
  integer perf_level interface is identical across them.
- `[PERF_LEVEL]` — optional integer 0–4 (see table above). Omit the
  argument entirely (bare `--enable-chip-swimlane`) for the level-4
  shorthand; omit the flag entirely for level 0 (disabled).

The flag sets `CallConfig::enable_chip_swimlane` to the chosen
level. The host then allocates the per-core / per-thread shared
region and publishes its base address through
`kernel_args.chip_swimlane_data_base`. AICore writes timing into
per-task WIP slots; the active Scheduler records dispatch/finish timestamps.
Per-task Scheduler timestamps are recorded only at level >= 2,
scheduler phase records only at level >= 3, and orchestrator phase
records only at level >= 4.

The JSON output `"chip_swimlane_level"` field is the captured perf_level:
`1` = AICore timing only, `2` = +Scheduler per-task dispatch/finish,
`3` = +scheduler phases, `4` = +orchestrator phases.

Chip-swimlane collection is disabled when `--rounds > 1` so benchmark
runs are not instrumented.

### 3.2 Output

The raw artifact lands under the per-task output prefix
(`CallConfig::output_prefix`, set by
`scene_test.py::_build_output_prefix` to
`outputs/<ClassName>_<case>_<YYYYMMDD_HHMMSS>/` for SceneTest
runs):

```text
<output_prefix>/
├── chip_swimlane_records.json     # raw runtime output
├── name_map_<case>.json     # optional func_id → name mapping
└── merged_swimlane.json     # Perfetto trace (added by converter)
```

Filenames are fixed (no per-file timestamp) — the directory is the
per-task uniqueness boundary.

For L3 runs, each forked ChipWorker writes below its own `rankN/dN` directory.
The filenames above are fixed, so N children sharing one `output_prefix` would
overwrite each other — the separation therefore covers **every** diagnostic that
writes below `output_prefix`, not just the swimlane:

```text
<output_prefix>/
├── rank0/d0/
│   ├── chip_swimlane_records.json    # --enable-chip-swimlane
│   ├── dispatch_identity.json        # always, whenever any diagnostic is on
│   ├── deps.json                     # --enable-dep-gen
│   └── scope_stats/                  # --enable-scope-stats
├── rank1/d0/
│   └── ...
└── l3_swimlane.json                  # cross-Rank trace (added by converter)
```

Here `rankN` is the logical ChipWorker index and `dN` is that worker's local
capture index. It is a storage-order suffix, not a globally comparable dispatch
ID. `dispatch_identity.json` records the parent scheduler identity: `run_id`,
`task_slot`, `group_index`, and `group_size`, plus the endpoint-local dispatch
and pipeline diagnostics. All members submitted through one
`submit_next_level_group` share `(run_id, task_slot)` and have distinct
`group_index` values. Individually submitted tasks do not share that identity;
the current postprocessor therefore retains local-capture-index pairing for
them and requires symmetric `dN` sets.

Automatic merging is limited to one same-host L3 Worker. NETWORK1/L4 is
rejected until the layout also carries a node namespace. Every Rank must expose
the same complete set of local capture indexes; the postprocessor refuses
asymmetric sets instead of guessing pairings.

Cross-Rank merging needs `--enable-chip-swimlane 4` on every Rank, because the
Host/Device clock anchors that level 4 collects are what put the Ranks on a
common timeline. A lower level still captures per Rank; the postprocessor then
converts each `rankN/dN` capture on its own relative timeline and says so.

`chip_swimlane_records.json` carries the raw records. **There are two
layers to be aware of:**

- **On-disk (raw, cycle domain).** What the host writes. Compact —
  per-stream tuples plus a small metadata block. Anyone reading the
  file directly with `json.load` sees this shape and only this shape.
- **Reader output (joined, µs domain).**
  `swimlane_converter.read_perf_data()` joins the on-disk streams, fills
  in `core_type` / `core_to_thread`, converts cycles to microseconds
  using `metadata.clock_freq_hz`, and returns the joined dict that every
  downstream consumer (Perfetto converter, `sched_overhead_analysis`,
  `deps_viewer`, in-test validator) reads. Always go through
  `read_perf_data`; never load `chip_swimlane_records.json` with raw
  `json.load` from new code.

#### On-disk schema

```jsonc
{
  "chip_swimlane_level": <1..4>,

  // Everything the python reader needs that isn't a per-record stream.
  "metadata": {
    "clock_freq_hz": <int>,            // cycle→µs factor. a2a3=50e6, a5=1e9.
    "num_cores": <int>,                // == len(core_types)
    "core_types": ["aic"|"aiv", ...],  // indexed by core_id
    "core_to_thread": [<int>, ...]     // optional; level >= 3 only
  },

  // Bulk task streams. Tuple column order is fixed.
  //   aicore_tasks: [core_id, task_token_raw, reg_task_id,
  //                  start_cycles, end_cycles]
  //   scheduler_tasks.records: [core_id, reg_task_id,
  //                             dispatch_cycles, finish_cycles]
  "aicore_tasks": [[...], ...],
  "scheduler_tasks": {
    "schema_version": 1,
    "producer": "<aicpu|aicore>",
    "records": [[...], ...]
  },

  // Producer-neutral per-Scheduler streams (level >= 3 only).
  "scheduler_records": {
    "schema_version": 1,
    "streams": [{
      "platform": "<a2a3|a5>",
      "runtime": "<host_build_graph|tensormap_and_ringbuffer>",
      "producer": "<aicpu|aicore>",
      "scheduler_id": <int>,
      "worker_id": <int>,
      "core_type": "<aicpu|aic|aiv>",
      "physical_core_id": "<int|null>",
      "capture": {"committed": <int>, "dropped": <int>, "truncated": <bool>},
      "records": [{"start_cycles": <int>, "end_cycles": <int>,
                   "loop_iter": <int>, "kind": <str>,
                   "tasks_processed": <int>, "task_id": "<int|null>"}],
      "metrics": [{"record_index": <int>, ...}]
    }]
  },

  // Orchestrator records (level >= 4 only).
  //   orch record:  {submit_idx, task_id, start_cycles, end_cycles}
  "aicpu_orchestrator_phases": [ [ {...}, ... ], ... ]   // level >= 4 only
}
```

All timestamps on disk are raw `get_sys_cnt` cycles (uint64). The
join key between `aicore_tasks` and `scheduler_tasks.records` is
`(core_id, reg_task_id)` — *not* `task_token_raw`, because SPMD
`block_num > num_cores` and MIX cluster spread can dispatch the same
`task_token_raw` to the same core multiple times. AICore is the
canonical producer of `task_token_raw`; the Scheduler producer stamps the
dispatch / finish timestamps and the per-core join token. Archived raw files
with the former `aicpu_tasks` array remain readable as `producer: "aicpu"`.

#### Reader output (µs domain)

After `read_perf_data()` joins the streams and converts to
microseconds, downstream code sees:

| Field | Meaning |
| ----- | ------- |
| `task_id` | Runtime task id (`TaskId::raw`); its high 32 bits are also exposed split off as `ring_id`, which is a ring index under `tensormap_and_ringbuffer` and an id space under `host_build_graph` |
| `func_id` | Kernel function id. Always `-1` on disk; resolved post-process from `deps.json::tasks[].kernel_ids[3]` (see `swimlane_converter.resolve_func_id_from_kernel_map`) |
| `core_id` / `core_type` | Physical core index and `"aic"` / `"aiv"` string |
| `start_time_us` / `end_time_us` / `duration_us` | AICore execution window in microseconds |
| `dispatch_time_us` | Scheduler timestamp when dispatch publication completed (filled at level >= 2) |
| `finish_time_us` | Scheduler timestamp when completion processing began (filled at level >= 2) |

Note: per-task records carry **no** fanout edges. Dependency arrows
come from a separate `deps.json` (dep_gen) joined at convert time —
see [§3.5](#35-dependency-arrows-from-dep_gen).

Phase records (per Scheduler stream, level >= 3 in raw
`scheduler_records`—also exposed through the legacy reader alias
`aicpu_scheduler_phases`—and level >= 4 for
`aicpu_orchestrator_phases[]`):

| Field | Meaning |
| ----- | ------- |
| `start_time_us` / `end_time_us` | Phase start / end timestamps in microseconds (reader-side cycle→µs conversion) |
| `phase` | Lowercase phase name. Scheduler: see the table below. Orchestrator: `orch_submit` — one record per `submit_task()` / `alloc_tensors()` call spanning its full `[start, end]` window. Legacy per-sub-step strings (`orch_sync` / `orch_alloc` / `orch_params` / `orch_lookup` / `orch_insert` / `orch_fanin`) may appear in old captures. |
| `loop_iter` (scheduler) / `submit_idx` (orchestrator) | Iteration / submit-call counter for the producing thread |
| `tasks_processed` (scheduler) | Number of tasks or blocks handled by the phase; `dummy_task` and `predicated_skip` record one task |
| `task_id` | Full runtime task id on orchestrator records and scheduler `dummy_task` / `predicated_skip` records |
| `pop_hit` / `pop_miss` (dispatch only) | Ready-queue pop deltas since the previous dispatch emit |

The raw scheduler record has a phase-tagged union: `dispatch` stores
`pop_hit` / `pop_miss`, while `dummy_task` and `predicated_skip` store the
32-bit `local_id` and `ring_id` components of their full task id. The Host
collector reconstructs the `task_id` JSON field.

Scheduler phase taxonomy — three role classes share one `phase`
field but render differently in Perfetto:

| Phase | Role | Lane | `tasks_processed` semantic |
| ----- | ---- | ---- | -------------------------- |
| `complete` | outer | sched (pid=2) | FIN'd subtasks + sub-block retires this iter |
| `async_poll` | outer | sched | async-wait completions resolved; zero means polling consumed CPU without completing work |
| `dispatch` | outer | sched | subtasks published this iter |
| `release` | outer | sched | deferred-release slots drained this iter |
| `dummy` | outer | sched | `dummy_ready_queue` entries handled this iter (explicit dummies and false-predicate tasks) |
| `early_dispatch` | outer | sched | blocks staged by speculative early-dispatch this pass |
| `drain` | outer | sched | blocks staged by this thread's global sync-start drain pass |
| `graph_prepare` | outer | sched | Graph Definition nodes expanded this pass |
| `resolve` | inner (TMR) | TMR sched sub-lane | consumers visited in `on_task_complete` |
| `resolve_standalone` | P-thread outer (HBG); rendered as `resolve` | HBG P sched lane | completed SPSC slots |
| `drain_prepare` | inner | sched, nested in `drain` | subtasks prepared for global sync-start publication |
| `drain_publish` | inner | sched, nested in `drain` | subtasks published during global sync-start staging |
| `dummy_task` | separate-lane | Worker View AICPU_N (pid=4) | one dummy entering `on_task_complete()`; full identity is in `task_id` |
| `predicated_skip` | separate-lane | Worker View AICPU_N (pid=4) | one real task retired inline after its dispatch predicate evaluated false; full identity is in `task_id` |

Fanin/fanout wiring is not a scheduler phase: it runs on the
orchestrator submit path, so it has no swimlane lane. Read its cost
from `g_orch_fanin_cycle` in the device-log orch breakdown (the
`fanin` line) instead.

Outer phases are mutually time-exclusive within an iter. In
`tensormap_and_ringbuffer`, the converter renders `resolve` on a sibling
`Sched_N` tid because it is time-contained by the outer `complete`/`dummy`
lane. In `host_build_graph`, standalone `resolve` stays beside `async_poll` and
`dummy` on the P thread's main scheduler lane. `drain_prepare` and
`drain_publish` remain on the scheduler lane and are time-contained by `drain`.
Separate-lane phases are routed to a different lane by the converter
(Worker View AICPU_N), so they never overlap visually with the sched lane
bars even when their timestamps fall inside an outer span.

On the HBG P thread, consecutive empty async-wait polls are compacted into one
`async_poll(0)` record. Its duration is the exact sum of time spent inside the
poll calls, anchored at the point where the aggregate is flushed; it is not a
wall-clock envelope over the intervening loop bookkeeping. The aggregate is
flushed before `resolve` or `dummy`, when a poll resolves work or reports an
error, and when P exits. This keeps polling cost visible without exporting one
record per spin. A non-zero `tasks_processed` counts every resolved async-wait
entry, including internal Graph nodes, rather than only host-submitted stream
tasks. The compacted record's `shared_at_start` snapshot comes from the first
poll in the aggregate, while `loop_iter` names the iteration that flushes the
aggregate. Because the displayed start timestamp is synthesized from summed
poll CPU time, neither field identifies one wall-clock iteration boundary.
The converter still emits the record's real `shared_at_end` snapshot on the
global ready-queue counter track; only the aggregate's start-side metadata has
the synthesized-timestamp caveat.

Legacy phases (`scan` / `poll` / `idle` / `fanout` / `prestage`)
are still parsed for old captures but current a2a3/a5 builds no
longer emit them. Renames: `fanout` → `resolve`, `prestage` →
`early_dispatch`. Removed: Scan/Poll (PR #1079 debug overlay).

On disk the sched records carry a `kind` field (string-encoded
phase name); the reader renames it to `phase` so downstream code
can keep a single discriminator name.

`core_to_thread[]` (level >= 3) maps `core_id` (array index) to the
scheduler thread index that retired that core's tasks (`-1` =
unassigned). On disk it lives under `metadata.core_to_thread`; the
reader hoists it to the top of the output dict.

### 3.3 Convert and view in Perfetto

`swimlane_converter` turns the raw records into a Perfetto trace
and produces a per-function task-execution summary:

```bash
# Auto-detects the latest outputs/*/chip_swimlane_records.json
python -m simpler_setup.tools.swimlane_converter

# Pin to a specific case + add func_id → name mapping
python -m simpler_setup.tools.swimlane_converter \
    outputs/<case>_<ts>/chip_swimlane_records.json \
    --func-names outputs/<case>_<ts>/name_map_<case>.json

# Custom output path
python -m simpler_setup.tools.swimlane_converter \
    outputs/<case>_<ts>/chip_swimlane_records.json -o my_trace.json

# Same-host L3: merge rankN/d0 captures onto one CLOCK_MONOTONIC timeline
python -m simpler_setup.tools.swimlane_converter \
    build_output/<case>/dfx_outputs --dispatch d0

# Prefer the parent group identity when Rank-local dN suffixes differ
python -m simpler_setup.tools.swimlane_converter \
    build_output/<case>/dfx_outputs --dispatch-id 17:5
```

For directory input, the default output is `dfx_outputs/l3_swimlane.json`.
Every Rank must be a level-4 capture under
`rankN/<dispatch>/`, with successful clock anchors and the same
`metadata.host_clock_domain_id`. The converter preserves real Rank start skew,
adds Rank-specific PID/name/flow namespaces, and reports clock uncertainty and
anchor-group observer overhead in trace metadata.

For new group captures, `--dispatch-id RUN_ID:TASK_SLOT` selects the common
parent DAG node and resolves each Rank's actual `dN` path through
`dispatch_identity.json`. SceneTest does this automatically. `--dispatch dN`
remains the compatibility selector for old captures and for independently
submitted per-Rank tasks; it fails if available sidecars show that the selected
paths belong to different parent groups.

Host-orchestrated level-4 runs retain their existing clock anchors. For
Device/AICPU orchestration, anchors are additionally enabled only when the
ChipWorker marks the capture with `CallConfig.capture_clock_anchors`, which it
does for an L3 chip-swimlane capture, at the common launch boundary before
collectors and kernels start. Both modes sample again after AICPU/AICore
execution completes. Existing single-card Device/AICPU level-4 captures
therefore keep their prior relative timeline and do not pay the new anchor cost.

`capture_clock_anchors` says only *what the runtime does* — sample the two
clocks — never why. Rank, group and merge are concepts of the layer above: the
platform runner that reads this flag has no notion of a Rank, and no runtime or
platform code parses the `rankN/dN` path. The two are deliberately separate
switches, because the directory is artifact separation that every diagnostic
needs while the anchors are consumed only by the swimlane reader. An L3 run with
`--enable-dep-gen` alone therefore gets its own `rankN/dN` directory and pays no
anchor cost.

**The opening anchor sits at a different point in each runtime**, because each
takes it at the earliest point preceding every device timestamp it records:

| Runtime | Opening anchor | Calibrated interval covers |
| ------- | -------------- | -------------------------- |
| `host_build_graph` | before Host orchestration (`host_phase_pool_arm`) | bind, H2D, and execution |
| `tensormap_and_ringbuffer` | before kernel launch (`start_shared_collectors_for_run`) | execution only |

Both close on `post_device_execution`. So the two runtimes' calibrated intervals
are not comparable in length, and a `host_build_graph` interpolation spans work
a `tensormap_and_ringbuffer` one does not. This does not affect
`max_uncertainty_ns`, which depends only on each anchor group's own sampling
RTT. The serialized position name `pre_host_orchestration` predates the
Device/AICPU case — read it as "start of the calibrated interval", not as a
claim about Host orchestration.

The default output depends on which input form was used, and `-o` overrides
either:

| Input | Default output |
| ----- | -------------- |
| a records file | `outputs/<case>_<ts>/merged_swimlane.json` |
| a `dfx_outputs` directory | `<dfx_outputs>/l3_swimlane.json` |

Open <https://ui.perfetto.dev/> and drag the file in. Both forms produce the
same lane structure — the directory form repeats it once per Rank under the
`rankN / <view>` process names. The trace contains:

- **Orchestrator** (pid=1) — per-submit `orch_submit` envelope
  blocks (level >= 4).
- **Scheduler** (pid=2) — per-iteration scheduler phase
  blocks coloured by `phase` (level >= 3). Outer phases appear as sibling bars
  on each scheduler thread's first `Sched_N` lane. TMR's nested `resolve`
  appears on an adjacent `Sched_N` sub-lane; HBG's standalone `resolve` stays
  on the P thread's first lane. `drain_prepare` and `drain_publish` nest within
  `drain`.
- **Scheduler View** (pid=3) — task-execution overlay using Scheduler
  dispatch/finish timestamps (level >= 2), with the same labels
  as Worker View.
- **Worker View** (pid=4) — one swim-lane per physical worker:
  - `AIC_N` — matrix cores (receive → kernel end from level >= 1)
  - `AIV_N` — vector cores (receive → kernel end from level >= 1)
  - `AICPU_N` — AICPU acting as worker; carries `dummy(...)` markers and
    ordinary task-named markers with `predicated_pass: false` in their Perfetto
    arguments. These 0.02 us pre-resolve markers represent one dependency-only
    completion entering `on_task_complete()` on AICPU N. It also carries
    `alloc` bars
    (from `alloc_tensors()` calls inline-completed by the dedicated
    orchestrator on the last AICPU runtime thread).
    Both are activities the AICPU performs as a worker, so they
    share the same lane tier as AIC/AIV. When `deps.json` is joined,
    dependency arrows involving dummy/predicated-skip/alloc DAG nodes anchor on these
    AICPU worker slices. The converter identifies dummy nodes from
    `deps.json` before consulting runtime timing records. If a dummy's
    scheduler record is missing, it warns and omits that Worker View bar
    instead of rendering it as `alloc`.
  AIC/AIV hover args keep both `kernel-duration-us` and
  `local_setup_us`; the old standalone setup preview bar is folded
  into the task bar.

`merged_swimlane.json` no longer emits separate `setup` X events.
For AIC/AIV tasks, the Worker View task bar starts at `receive_time_us`
and ends at `end_time_us`; the kernel-only duration remains available as
`kernel-duration-us`, and the receive→start setup interval remains
available as `local_setup_us` in the task's hover args.

**Task labeling (AICore View and AICPU View) depends entirely on
whether a `deps.json` is present** (see
[§3.5](#35-dependency-arrows-from-dep_gen)):

- **With `deps.json`** — each task shows `func_name(rXtY)` (or
  `func_<id>(rXtY)` when no name map), and dependency arrows are
  drawn in the task views.
- **Without `deps.json`** — the host never records `func_id` (it's
  `-1` on disk), so the converter cannot tell tasks apart by
  function. Every task in **both** views is labeled `task(rXtY)` —
  distinguished only by id — and no arrows are drawn (the converter
  prints a one-line hint). Re-run with `--enable-dep-gen` (or join an
  existing `deps.json`) to recover names and arrows.

`swimlane_converter` does not run the scheduler-overhead deep-dive. Capture
`deps.json` and `chip_swimlane_records.json` in separate runs, then invoke
`sched_overhead_analysis` explicitly as described in the
[tool documentation](../../simpler_setup/tools/README.md#sched_overhead_analysis).

The scheduler-budget parser counts every mutually exclusive outer phase and
standalone HBG P-thread `resolve` bars. It excludes only `resolve` records whose
timestamps are contained by a TMR `complete` or `dummy` parent, preventing the
nested TMR work from being counted twice.

### 3.4 Adding human-readable names

Lane labels degrade in two steps:

| What's available | Label | Distinguishable? |
| ---------------- | ----- | ---------------- |
| No `deps.json` (no `--enable-dep-gen`) | `task(rXtY)` | By id only — `func_id` is unresolved |
| `deps.json`, no name map | `func_<id>(rXtY)` | By function id |
| `deps.json` + name map | `QK(rXtY)` | By human name |

So a readable trace needs **both** a `deps.json` (to resolve
`func_id`; see [§3.5](#35-dependency-arrows-from-dep_gen)) **and** a
name map. To get readable lane labels, add a `name` field to your
CALLABLE spec:

```python
@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPagedAttention(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/orch.cpp",
            "function_name": "build_paged_attention_graph",
            "name": "PagedAttn",                          # optional
            "signature": [D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {"func_id": 0, "name": "QK", "source": "...", "core_type": "aic"},
            {"func_id": 1, "name": "SF", "source": "...", "core_type": "aiv"},
            {"func_id": 2, "name": "PV", "source": "...", "core_type": "aic"},
        ],
    }
```

SceneTest extracts this into `<output_prefix>/name_map_<case>.json`
and passes it to `swimlane_converter` automatically. See
[profiling-name-map.md](profiling-name-map.md) for the full
schema and L3 example.

### 3.5 Dependency arrows from dep_gen

Swimlane records carry **timing only**; they do not embed per-task
fanout. The device hot path deliberately omits it (see the
`ChipSwimlaneAicpuTaskRecord` comment and PR #863 — a per-task ~1 KB
GM store + a linked-list walk on the scheduler's critical fanin tail
was the price). Dependency arrows in the Perfetto view come from
`deps.json`, the dep_gen artifact, joined at post-process time by
`swimlane_converter`.

Two artifacts, one join:

| File | Producer | What it carries |
| ---- | -------- | --------------- |
| `deps.json` | `--enable-dep-gen` (a [`dep_gen`](dep-gen.md) run) | The static task graph for one topology / case |
| `chip_swimlane_records.json` | `--enable-chip-swimlane` | Per-task / per-phase timing for one run |
| `merged_swimlane.json` | `swimlane_converter` | Perfetto trace = timing joined to the graph |

**Co-capture workflow (functional debugging and CI smoke):**

```bash
python test_my_case.py --platform a2a3 \
  --enable-dep-gen --enable-chip-swimlane
```

Both artifacts land under the same `<output_prefix>/`; the
converter auto-detects `deps.json` and emits flow arrows. This is the
convenient path for CI smoke and one-off functional debugging. Do not use
co-captured timing for strict scheduler-overhead measurement because dep_gen
adds per-submit work to the measured run.

**Split workflow (two launches, required for strict scheduler-overhead
measurement):**

```bash
# Once per topology — produces deps.json.
python test_my_case.py --platform a2a3 --enable-dep-gen

# Any number of perf-measurement runs against the same topology —
# each produces its own chip_swimlane_records.json, all joined to the
# captured graph.
python test_my_case.py --platform a2a3 --enable-chip-swimlane 4
python -m simpler_setup.tools.swimlane_converter \
    outputs/<case_ts>/chip_swimlane_records.json \
    --deps-json outputs/<case_dep_ts>/deps.json
```

Use this when:

- You're measuring scheduler overhead and need the swimlane timing run free
  from dep_gen instrumentation.
- The same topology is being measured under several configurations
  (one `dep_gen` capture amortizes across N swimlane runs).
- A workload is so large that the dep_gen replay validation gate
  would dominate the swimlane run time.

**Low-distortion workflow (minimal capture, names recovered
offline):** When you want the least possible perturbation — skip the
`dep_gen` replay overhead on the measured run, and/or use a low
swimlane perf_level (e.g. level 1, AICore timing only) — run the perf
capture *without* `--enable-dep-gen`. The resulting trace labels every
task `task(rXtY)` (no `func_id`, no arrows). To recover names and
arrows afterward, take a `deps.json` from a separate `dep_gen` capture
of the same topology, drop it next to your `chip_swimlane_records.json`
(or point `--deps-json` at it), and re-run the converter:

```bash
# Low-overhead perf run — no dep_gen, low level.
python test_my_case.py --platform a2a3 --enable-chip-swimlane 1

# Separately (once per topology), capture the graph.
python test_my_case.py --platform a2a3 --enable-dep-gen

# Join offline: copy the deps.json in, then re-run the converter.
cp outputs/<case_dep_ts>/deps.json outputs/<case_perf_ts>/deps.json
python -m simpler_setup.tools.swimlane_converter \
    outputs/<case_perf_ts>/chip_swimlane_records.json \
    --func-names outputs/<case_perf_ts>/name_map_<case>.json
```

The converter is a pure post-processor — re-running it against the
same raw records with a `deps.json` now present upgrades the labels
from `task(rXtY)` to real names and adds the dependency arrows,
without re-running the workload.

When `--deps-json` is omitted **and** the converter cannot find a
sibling `deps.json` next to `chip_swimlane_records.json`, the trace is
emitted without flow events (correct, just no arrows) and the
converter prints:

```text
Flow events: 0 (no deps.json — re-run dep_gen and pass --deps-json to add arrows)
```

That's the rerun breadcrumb — keep an eye on it, it's the signal
that something dropped on the way from dep_gen to converter.

**SPMD dependency arrows.** For logical tasks with `block_num > 1`,
dependency / `hb_violation` flows connect via **anchor pairing** on
the Worker View and Scheduler View task lanes — there is no dedicated
`SPMD (block-level)` track.

Each view independently selects one SPMD anchor for every
`(func_id, task_id)` group. The Worker View chooses the earliest visible
kernel slice: `receive_time_us` when present, including the valid value `0`,
or `start_time_us` for archived records without a receive timestamp. The
Scheduler View independently chooses the earliest visible AICPU slice by
`dispatch_time_us`. Equal start times are resolved by the smaller `core_id`.
The two views can therefore select different physical subtask records. Within
each view, dependency and `complete` arrows use that view's selected anchor.
Dependency flows connect the visual starts of the selected source and
destination bars. The source completion timestamp still determines whether
the flow is named `dependency` or `hb_violation`; it does not become the flow
start because a completion later than the destination bar start would create
a reverse-time flow that Perfetto cannot display.

The grouping includes both the function identity and the logical `task_id`
(ring/local id), so MIX tasks that share a `task_id` across AIC/AIV functions
keep separate anchors. The converter does not draw one arrow per subtask
instance.

**`complete` arrows.** Like the dependency mirror, the per-task
`complete` flow (task → the pid=2 `complete` phase that observed its
last subtask FIN) is drawn from **both** task views using their independently
selected anchor rows. The Worker View source anchors on the kernel slice
(`end_time_us`), and the Scheduler View source anchors on the AICPU
`finish_time_us`. Both arrows land on the identical pid=2 endpoint (thread +
timestamp), so clicking the task in either view surfaces the arrow without
changing completion attribution. The Scheduler View arrow is skipped when
there is no visible AICPU bar.

Non-SPMD tasks (including MIX multi-slot kernels with `block_num == 1`)
keep every subtask row as an endpoint (N×N pairing unchanged).

For each logical `(pred, succ)` edge from `deps.json`, the converter
emits flows between the Cartesian product of pred/succ anchor rows
(`|pred_anchors| × |succ_anchors|`), not a per-subtask crossbar.
Kernel tasks anchor on AIC/AIV task rows; dummy and alloc DAG nodes
anchor on the AICPU worker slices that represent those activities.

**SPMD lane labels.** Logical SPMD tasks append `_spmd` before the
`(rXtY)` suffix unless the function name already contains `spmd`
(case-insensitive), e.g. `v_proj_spmd(r2t10)` vs `SPMD_WRITE_AIV(t0)`.

Flow events carry `input_task_count` / `output_task_count` (SPMD
`block_num`) to annotate fan degree. These arrows visualize **block-level**
`deps.json` edges on representative subtasks — they do **not** imply
runtime per-instance dependency resolution.

**What you do NOT need to script:**

- Pairing input shape / RNG seed across the two launches — `deps.json`
  is graph-shaped, not instance-shaped; two runs of the same case
  with the same `CASES[...]` entry share the same graph by
  construction.
- Running dep_gen ahead of every swimlane run — the graph is stable
  per topology; one capture is enough until the test class changes.

## 4. Capabilities

What the swimlane shows:

- **Per-task wall-clock placement.** Where each task ran on which
  AICore, with `start_time_us` / `end_time_us` / `duration_us` in
  microseconds (converted from device cycles).
- **Dispatch and finish overhead.** `dispatch_time_us` and
  `finish_time_us` come from AICPU, so the gap between
  `dispatch_time_us` and `start_time_us` is the AICPU→AICore
  hand-off latency, and the gap between `end_time_us` and
  `finish_time_us` is the FIN-observation latency.
- **Dependency chains.** When `deps.json` from a paired or prior
  `dep_gen` run is available, `swimlane_converter` emits flow events
  so Perfetto draws arrows between predecessor and successor tasks
  — see [§3.5](#35-dependency-arrows-from-dep_gen). Without
  `deps.json` the trace is correct but unarrowed. For SPMD tasks,
  dependency arrows independently use each view's earliest visible slice per
  `(func_id, task_id)` group as the anchor.
- **Scheduler-loop time decomposition.** Per-iteration AICPU
  phase records show how long the scheduler spent in recorded work phases;
  idle is recovered
  from the gap between records.
- **Orchestrator overhead breakdown.** Per-submit envelope
  records (`orch_submit`) pin "which submit is slow"; cumulative
  cycle counts in the cold-path device log (`g_orch_*_cycle`)
  cover the per-sub-step breakdown for "which sub-step dominates".

## 5. Design Highlights

### 5.1 Common interfaces

`kernel_args.chip_swimlane_data_base` is the single device-side handle
host publishes for the run. The shared region carries a fixed
`ChipSwimlaneDataHeader` plus per-core / per-thread state (same struct
shape on both architectures):

```text
ChipSwimlaneDataHeader                               (host init, device R/W)
├── queues [MAX_AICPU_THREADS][PROF_READYQUEUE_SIZE]
├── queue_heads / queue_tails  (per-thread)
├── num_cores / chip_swimlane_level                   (host writes at init)
├── num_sched_phase_threads / num_orch_phase_threads  (AICPU writes at phase
├── num_phase_cores / core_to_thread[]                 init; host gates on
│                                                      the two counts)
└── backpressure                (DfxBackpressureHeader)

Every pool below is the same 192B shape — one ChipSwimlaneActiveHead (64B)
plus one ChipSwimlaneFreeQueue (128B). Only the buffer payload type differs,
which is what lets the drain machinery stay polymorphic:

head       {current_buf_ptr, current_buf_seq,     single 64B cache line: the
            total_record_count,                   producer's active buffer
            dropped_record_count}                 plus its accounting
free_queue {buffer_ptrs[PROF_SLOT_COUNT],         SPSC ring — host pushes
            head, tail}                           recycled buffers, AICPU pops

ChipSwimlaneAicpuTaskPool[num_cores]                    (AICPU writes)
└── ChipSwimlaneAicpuTaskBuffer × PLATFORM_PROF_BUFFERS_PER_CORE
    └── ChipSwimlaneAicpuTaskRecord records[PLATFORM_PROF_BUFFER_SIZE]
                                                        (1000 records, 32B each)

ChipSwimlaneAicoreTaskPool[num_cores]                   (AICore writes; AICPU
└── ChipSwimlaneAicoreTaskBuffer                         rotates and bumps
    × PLATFORM_AICORE_BUFFERS_PER_CORE                   current_buf_seq, which
    └── ChipSwimlaneAicoreTaskRecord                     AICore dcci-polls per
        records[PLATFORM_AICORE_BUFFER_SIZE]             task to detect)
                                                        (1024 records, 32B each)

ChipSwimlaneAicpuSchedPhasePool[MAX_AICPU_THREADS]      (one per scheduler
└── ChipSwimlaneAicpuSchedPhaseBuffer                    thread; buffers only
    × PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD             for live threads)
    └── ChipSwimlaneAicpuSchedPhaseRecord
        records[PLATFORM_PHASE_RECORDS_PER_THREAD]      (16384 records, 64B each)

ChipSwimlaneAicpuOrchPhasePool[MAX_AICPU_THREADS]       (orchestration is
└── ChipSwimlaneAicpuOrchPhaseBuffer                     single-threaded, so
    × PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD              only ordinal 0 ever
    └── ChipSwimlaneAicpuOrchPhaseRecord                 gets a producer)
        records[PLATFORM_PHASE_RECORDS_PER_THREAD]      (16384 records, 32B each)
```

The pool-state array is sized `PLATFORM_MAX_AICPU_THREADS` on both phase
kinds because AICPU's offset stride is fixed at that width; only the first
`aicpu_thread_num` states ever get buffers seeded.

Task records are identical across architectures:

- `ChipSwimlaneAicpuTaskRecord` — per-task AICPU-owned fields (dispatch_time,
  finish_time, reg_task_id); 20 B logical, `aligned(32)` so two records share
  a cache line. `reg_task_id` is the join key against the matching AICore
  record. `core_type` comes from the static per-core table
  `ChipSwimlaneCollector::set_core_types` publishes, and `func_id` is resolved
  post-process by `swimlane_converter.py` from deps.json's `kernel_ids[]` —
  neither is carried in the record.
- `ChipSwimlaneAicoreTaskRecord` — slim AICore-only record (start_time,
  end_time, task_token_raw, reg_task_id, receive_to_start_cycles), 32 bytes;
  AICore writes one per task into its currently-active per-core buffer.
  `reg_task_id` is the join key; `task_token_raw` is identity only.

Both architectures use split phase streams:

- `ChipSwimlaneAicpuSchedPhaseRecord` (64 B) — one record per **emitted
  phase**, not per scheduler iteration: a single iteration routinely emits
  several (e.g. Complete, AsyncPoll, Dispatch, Release, plus Resolve).
  `ChipSwimlaneSchedPhaseKind` spans the outer phases
  (Complete, Dispatch, Release, Dummy, EarlyDispatch, AsyncPoll, Drain,
  GraphPrepare, ResolveStandalone), TMR's inner Resolve, the inner drain phases
  (DrainPrepare, DrainPublish), and
  the separate-lane markers (DummyTask, PredicatedSkip) — see §3.2 for how
  each is rendered. Carries loop_iter + tasks_processed + pop_hit /
  pop_miss deltas and queue-depth snapshots.
- `ChipSwimlaneAicpuOrchPhaseRecord` (32 B) — per-submit orchestrator
  envelope; task_id + submit_idx + start/end.

`swimlane_converter` consumes the shared shape and produces the same
output JSON on both architectures. The orch stream replaces the per-sub-step
records folded into ORCH_SUBMIT; there is no separate shared-memory
aggregate. The run-window envelope is emitted to device log via
`LOG_INFO "orch_start=… orch_end=… orch_cost=…"`.

**Producer/consumer protocol on AICore (AICore-as-producer with rotation).**
AICore writes a slim `ChipSwimlaneAicoreTaskRecord` into its currently-active per-core
`ChipSwimlaneAicoreTaskBuffer` at `records[slot_within_buf++]`. The active buffer is
published via a per-core `ChipSwimlaneActiveHead` cache line (`current_buf_ptr` +
`current_buf_seq` + counters); AICore `dcci`'s it per task — cheap relative
to the baseline `dcci(payload, ENTIRE_DATA_CACHE)` it already pays per
task. AICPU drives rotation: immediately before each `write_reg(DATA_MAIN_BASE)`
for task `K`, if `K % PLATFORM_AICORE_BUFFER_SIZE == 0`, AICPU enqueues
the current buffer to the per-thread ready queue (kind `AicoreTask`),
pops the next from `ChipSwimlaneAicoreTaskPool::free_queue`, and bumps
`ChipSwimlaneActiveHead::current_buf_seq`. AICore detects the bumped seq on
its next task's `dcci`, refreshes its local cache, and resets its slot
counter to 0.

**Race safety.** The runtime's completion-before-dispatch invariant
guarantees all tasks `< K` have FIN'd before AICPU dispatches `K`, so
by the time AICPU enqueues the old buffer, AICore has finished writing
(and `dcci+dsb`'d) records for all those tasks. No spin-wait, no
cross-direction read on the hot path.

**Sizing.** `PLATFORM_AICORE_BUFFER_SIZE = 1024` (power of two, modulo
lowers to AND) and `PLATFORM_AICORE_BUFFERS_PER_CORE = 4` (1 active +
3 recycled). Host-side `BufferPoolManager` drains full buffers through
the collector, returns them via done → replenish → recycled lanes, and
the owning drain shard refills the device free queue from that lane. The
replenish thread also keeps recycled lanes above their host-side watermarks
by batched allocation, but it never writes device free queues. These
watermarks are steady-state low-water marks: AICPU-task keeps half of the
init-seeded surplus per shard, while AICore-task has no init surplus and only
keeps a minimal reserve batch. Session length is bounded only by how fast the
host closes that loop — not by the per-core buffer sum.

**Measured impact.** Hardware bench on a2a3 paged_attention_unroll
Case1 with swimlane=4: rotation design delivers sched -4 µs / orch -19 µs
vs the upstream/main baseline, comparable to the no-rotation predecessor
(which had this PR's earlier commit; the rotation adds about 3 µs
sched overhead per session as price for unbounded session length).

### 5.2 a2a3 — shared-memory streaming

`halHostRegister` maps device memory into host virtual address
space so the host can read device buffers directly.
`ChipSwimlaneCollector` runs split mgmt threads and collector shards on top of a
[`BufferPoolManager<ChipSwimlaneModule>`](../../src/common/platform/include/host/buffer_pool_manager.h):
drain/refill shards poll SPSC ready queues and refill free queues from
shard-local recycled lanes **while kernels are still executing**. Collector
shards drain the host hand-off queues into `on_buffer_collected`, then the
replenish thread routes done buffers to same-kind lanes below their recycled
watermarks before allocating any remaining top-up.

`ChipSwimlaneModule` declares four buffer kinds going through one ready
queue per AICPU thread:

- **kind 0** `AicpuTask`        — per-core `ChipSwimlaneAicpuTaskBuffer` (AICPU writes).
- **kind 1** `AicpuSchedPhase`  — per-thread `ChipSwimlaneAicpuSchedPhaseBuffer` (AICPU writes).
- **kind 2** `AicpuOrchPhase`   — per-thread `ChipSwimlaneAicpuOrchPhaseBuffer` (AICPU writes).
- **kind 3** `AicoreTask`       — per-core `ChipSwimlaneAicoreTaskBuffer` (AICore writes,
  AICPU enqueues on rotation).

Each `ReadyQueueEntry::kind` carries the discriminator. This is the
only multi-kind module in the current framework — PMU and ArgsDump
are single-kind.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ ChipSwimlaneCollector          │               │ AICPU + AICore           │
│                          │               │                          │
│ initialize(prefix)       │  alloc +      │ AICore on task end:      │
│   rtMalloc + halRegister │──register────>│   write slim record into │
│   pre-fill free queues   │              │   AicoreTaskBuffer (kind │
│   for kinds 0/1/2/3      │               │   3); AICPU rotates it   │
│                          │               │                          │
│ start(tf)                │               │ AICPU on FIN:            │
│   ┌────────────────────┐ │ SPSC ready    │   commit AicpuTask       │
│   │ drain/refill shard │ │ queues        │   record (kind 0):       │
│   │ + replenish thread │ │<──4 kinds────<│   dispatch / finish /    │
│   │   poll ready queue │<┼──multiplexed──│   reg_task_id; rotate    │
│   │   refill freeQ     │─┼──free queue──>│   buffer when full       │
│   └────────────────────┘ │               │ AICPU scheduler thread:  │
│   ┌────────────────────┐ │               │   per emitted phase:     │
│   │ collector shard    │ │               │   SchedPhaseRecord       │
│   │   reads via host   │ │ shared mem    │   (kind 1). Per submit:  │
│   │   mapping; copies  │<┼──mapping─────<│   write OrchPhaseRecord  │
│   │   to host vectors  │ │               │   (kind 2).              │
│   └────────────────────┘ │               │                          │
│ stop()                   │               │                          │
│   join mgmt → collectors │               │                          │
│ read_phase_header_metadata()             │                          │
│ reconcile_counters()     │               │                          │
│ export_swimlane_json()   │               │                          │
│   → chip_swimlane_records.json │               │                          │
└──────────────────────────┘               └──────────────────────────┘
```

**Lifecycle** (`device_runner.cpp`):

```text
init_chip_swimlane()
  chip_swimlane_collector_.initialize(num_aicore, ..., output_prefix_)
  kernel_args_.args.chip_swimlane_data_base =
      chip_swimlane_collector_.get_chip_swimlane_setup_device_ptr()
  kernel_args_.args.chip_swimlane_aicore_rotation_table =
      chip_swimlane_collector_.get_aicore_ring_addr_table_device_ptr()
start(tf)                          ← spawn split mgmt + collector shards
launch AICPU / AICore
rtStreamSynchronize
stop()                             ← join mgmt/replenish → join collectors
read_phase_header_metadata()       ← single-shot read of the
                                     core→thread mapping
reconcile_counters()               ← two-bucket accounting per pool
                                     (PERF, SCHED_PHASE, ORCH_PHASE):
                                     collected + dropped == device_total;
                                     any non-zero current_buf_ptr over a
                                     non-empty buffer is a flush bug
export_swimlane_json()             ← writes <output_prefix>/chip_swimlane_records.json
finalize(unregister, free)
```

[`ChipSwimlaneCollector`](../../src/common/platform/include/host/chip_swimlane_collector.h)
on a2a3 inherits from
[`profiling_common::ProfilerBase<ChipSwimlaneCollector, ChipSwimlaneModule>`](../../src/common/platform/include/host/profiler_base.h):
the base class owns split mgmt threads, collector shards, and the
`BufferPoolManager<ChipSwimlaneModule>` they share. `ChipSwimlaneCollector`
supplies the L2-specific pieces — the `ChipSwimlaneModule` trait
(notably `kBufferKinds = 4` and `kind_of()`), `initialize` that
allocates and pre-fills all four kinds of free queues, an
`on_buffer_collected` callback that branches on `info.type` across
`AICPU_TASK` / `AICPU_SCHED_PHASE` / `AICPU_ORCH_PHASE` / `AICORE_TASK`
to copy into the right per-core or per-thread vector, plus
`read_phase_header_metadata` /
`reconcile_counters` / `export_swimlane_json` / `finalize`. The
mgmt/collector threading and `Module` trait pattern are shared with
PMU and ArgsDump — see
[profiling-framework.md](profiling-framework.md) for the
framework reference.

### 5.3 a5 — same framework, host-shadow transport

a5's `ChipSwimlaneCollector` derives from
`ProfilerBase<ChipSwimlaneCollector, ChipSwimlaneModule>` and uses the same
framework abstractions as a2a3, including the same split mgmt +
collector shard shape (`kMaxCollectorThreads` =
`PLATFORM_MAX_AICPU_THREADS`, i.e. 5 on a5 vs 4 on a2a3, capping the
shard arrays; the live drain/collector count is
`min(aicpu_thread_num, kMaxCollectorThreads)`). The
behavioral deviation from §5.2 is the **transport channel**: a5 has no
`halHostRegister`, so each device buffer is paired with a
host-shadow `malloc()` and the mgmt loop synchronizes the two via
`profiling_copy.h` (`rtMemcpy` onboard, plain `memcpy` in sim).

What is **stable** per core is the *head slot address*, not the buffer. The
host publishes a `uint64_t[num_aicore]` rotation table through
`KernelArgs::chip_swimlane_aicore_rotation_table`; `KERNEL_ENTRY` indexes it by
`block_idx` and hands the slot to `set_chip_swimlane_aicore_head_slot()`. AICPU's
`chip_swimlane_aicpu_init` fills each slot with the address of that core's
`ChipSwimlaneAicoreTaskPool::head`. AICore therefore holds one address for the
whole run, but what it *finds* there rotates: it dcci-polls the head per task
and re-reads `current_buf_ptr` whenever `current_buf_seq` bumps (§5.1). AICPU
task-buffer rotation is separately internal to
`chip_swimlane_aicpu_complete_task` when `records[count]` hits
`PLATFORM_PROF_BUFFER_SIZE`. The runtime `Handshake` carries no
profiling fields.

The framework's `MemoryOps` therefore carries five callbacks on
a5 (`alloc` / `reg` / `free_` / `copy_to_device` /
`copy_from_device`); the mgmt loop mirrors the entire shm region
(`ChipSwimlaneDataHeader` + the per-core task pools + the per-thread
sched- and orch-phase pools)
device → host at the top of every tick, then pushes back only the
fields host actually modified (advanced `queue_heads[q]`, refilled
`free_queue.tail` and `buffer_ptrs[slot]`) via
`BufferPoolManager::write_range_to_device`. The bulk
`mirror_shm_to_device` is deliberately **not** called from the mgmt
loop: it would race with AICPU writes to device-only fields
(`head.current_buf_ptr`, `head.current_buf_seq`, `head.total/dropped`
counters, `queue_tails`, `free_queue.head`, and the header's
`num_{sched,orch}_phase_threads` / `num_phase_cores` / `core_to_thread[]`)
and roll them back to
whatever the host shadow held at the start of the tick. Per-buffer
payloads (`ChipSwimlaneAicpuTaskBuffer` /
`ChipSwimlaneAicpuSchedPhaseBuffer` / `ChipSwimlaneAicpuOrchPhaseBuffer`)
are pulled on demand inside `ProfilerAlgorithms::process_entry` after
a popped ready-entry resolves to its host shadow. `BufferPoolManager`'s
`release_owned_buffers` canonicalizes carved sub-buffers back to the
registered allocation block before calling the collector's `release_fn`;
paired host shadows are released later by `clear_mappings()` or on init
rollback by `release_all_owned()`.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ ChipSwimlaneCollector          │               │ AICPU + AICore           │
│   : ProfilerBase<...>    │               │                          │
│                          │               │                          │
│ initialize()             │  alloc + reg  │ AICore on task end:      │
│   rtMalloc shm           │──+ shadow────>│   dcci the head slot,    │
│   per-core AicpuTaskBuf  │   memset 0    │   write a slim record    │
│   per-core AicoreTaskBuf │   + push 0s   │   into the active        │
│   per-thread Sched/Orch  │               │   AicoreTaskBuffer       │
│     PhaseBuf             │               │                          │
│   register_mapping(s)    │               │ AICPU on FIN:            │
│   set_memory_context     │               │   commit AicpuTask       │
│                          │               │   record into records[]  │
│ start(thread_factory)    │               │                          │
│   mgmt_thread starts     │               │ AICPU per-thread flush   │
│   poll_thread starts     │               │   on exit: enqueue       │
│                          │               │   current_buf_ptr →      │
│ mgmt every 10us tick:    │               │   ready_queue            │
│   copy_from_device(shm)  │<──memcpy─────<│                          │
│   for each ready entry:  │               │                          │
│     copy buf from device │<──memcpy─────<│                          │
│     resolve host ptr     │               │                          │
│     push to host ready_q │               │                          │
│   advance queue_heads,   │               │                          │
│     refill free_queues   │               │                          │
│   write_range_to_device  │──memcpy──────>│                          │
│     for each modified    │               │                          │
│     field                │               │                          │
│                          │               │                          │
│ poll thread:             │               │                          │
│   wait_pop_ready          │               │                          │
│   on_buffer_collected →  │               │                          │
│     copy_perf/phase      │               │                          │
│   notify_copy_done       │               │                          │
│                          │               │                          │
│ rtStreamSynchronize      │               │                          │
│ stop()                   │               │                          │
│   join mgmt + poll       │               │                          │
│ read_phase_header_meta   │               │                          │
│ reconcile_counters       │               │                          │
│   sanity-check leftovers │               │                          │
│   + 2-bucket cross-check │               │                          │
│ export_swimlane_json()   │               │                          │
│ finalize(free)           │               │                          │
└──────────────────────────┘               └──────────────────────────┘
```

**Lifecycle** (`device_runner.cpp`):

```text
init_chip_swimlane()
  chip_swimlane_collector_.initialize(num_aicore, ..., output_prefix_)
  kernel_args_.args.chip_swimlane_data_base = chip_swimlane_collector_.get_chip_swimlane_setup_device_ptr()
  kernel_args_.args.chip_swimlane_aicore_rotation_table =
      chip_swimlane_collector_.get_aicore_ring_addr_table_device_ptr()
chip_swimlane_collector_.start(thread_factory)   ← mgmt + poll threads
launch AICPU / AICore
rtStreamSynchronize
chip_swimlane_collector_.stop()                  ← join mgmt + poll, drain final batch
chip_swimlane_collector_.read_phase_header_metadata()
chip_swimlane_collector_.reconcile_counters()    ← sanity-check + 2-bucket cross-check
chip_swimlane_collector_.export_swimlane_json()
chip_swimlane_collector_.finalize()
```

[`ChipSwimlaneCollector`](../../src/common/platform/include/host/chip_swimlane_collector.h)
on a5 inherits the same CRTP base
([`profiling_common::ProfilerBase`](../../src/common/platform/include/host/profiler_base.h))
as a2a3 and parameterizes
[`BufferPoolManager`](../../src/common/platform/include/host/buffer_pool_manager.h)
with `ChipSwimlaneModule` (`kBufferKinds = 4`). The collector source is
shared — `src/common/platform/{include,shared}/host/` — so the only
a5-specific glue is the 5-callback `MemoryOps` and the per-tick shm mirror.

a5's per-thread AICPU flush hooks (`chip_swimlane_aicpu_flush` /
`chip_swimlane_aicpu_flush_phase_buffers`) are the only data path on the
records side — host never reads from `current_buf_ptr` to recover
records. `reconcile_counters` is purely passive: it logs an error if
any `current_buf_ptr` is non-zero with a non-empty buffer (a
device-flush bug), then runs the two-bucket cross-check
`collected + dropped == device_total` per pool (PERF, SCHED_PHASE,
ORCH_PHASE), same shape as a2a3.

### 5.4 a2a3 vs a5 at a glance

| Aspect | a2a3 | a5 |
| ------ | ---- | -- |
| Task record | `ChipSwimlaneAicpuTaskRecord` (32 B) + `ChipSwimlaneAicoreTaskRecord` (32 B) | identical |
| Phase record | `ChipSwimlaneAicpuSchedPhaseRecord` (64 B) + `ChipSwimlaneAicpuOrchPhaseRecord` (32 B) | identical |
| AICore head-slot rotation protocol | identical | |
| AICPU commit on FIN | identical | |
| Buffer model | rotating pool (free + ready queues) per kind | identical |
| Ready queue | per-AICPU-thread, multiplexes 4 kinds via `ReadyQueueEntry::kind` | identical |
| Host threads | split mgmt + collector shards, streams during execution | same split mgmt + collector shards (5 = `PLATFORM_MAX_AICPU_THREADS` vs a2a3's 4) |
| Host-class shape | `ProfilerBase<ChipSwimlaneCollector, ChipSwimlaneModule>` (`kBufferKinds = 4`) | identical — one shared collector under `src/common/platform/` |
| Host transport | `halHostRegister` shared memory | host-shadow `malloc` + per-tick `rtMemcpy`/`memcpy` |
| `MemoryOps` callbacks | 3 (`alloc`, `reg`, `free_`) | 5 (+ `copy_to_device`, `copy_from_device`) |
| `reconcile_counters` | passive cross-check (collected + dropped == device_total) | identical |
| Lifecycle | `initialize` → `start` → `stop` → `read_phase_header_metadata` → `reconcile_counters` → `export_swimlane_json` → `finalize` | identical |

## 6. Overhead

chip swimlane is opt-in and zero-overhead when disabled — without
`--enable-chip-swimlane` neither host nor device allocates the L2
perf shared region and the timing-write code paths are skipped.

When enabled, the dominant per-task overhead is:

- `get_sys_cnt()` reads at task start / end on AICore.
- Two cache-line writes into the WIP slot.
- The AICPU commit on FIN, which copies the WIP record into the
  ring buffer plus a few metadata fields.

Phase-record overhead, same on both architectures:

- at `--enable-chip-swimlane >= 3` — one 64 B
  `ChipSwimlaneAicpuSchedPhaseRecord` per **emitted phase**, so a scheduler
  iteration that does several kinds of work costs several records; idle
  iterations emit none.
- at `--enable-chip-swimlane >= 4` only — one 32 B
  `ChipSwimlaneAicpuOrchPhaseRecord` per `submit_task()`. Level 3 captures
  no orchestrator records at all (`ChipSwimlaneLevel::ORCH_PHASES` gates
  both the pool allocation and the device-side write).

Both architectures drain buffers concurrently with execution through the
ProfilerBase mgmt/collector pipeline; both a2a3 and a5 use split mgmt plus
collector shards for this profiler, capped at `PLATFORM_MAX_AICPU_THREADS`
(a5 5, a2a3 4). a5
additionally pays per-buffer `rtMemcpy`/`memcpy` round-trips to keep the
host shadow in sync, which overlap with device execution.

`--rounds > 1` disables chip-swimlane collection so the steady-state
benchmark is not perturbed.

## 7. Limitations

### 7.1 a2a3

- Records can be lost on device when both the per-core / per-thread
  free queue and the host's recycled pool are empty for too long.
  AICPU increments `dropped_record_count` and continues; the host's
  `reconcile_counters()` reports `collected + dropped == total` per
  pool. If `dropped > 0`, raise `PLATFORM_PROF_BUFFERS_PER_CORE` /
  `PLATFORM_PROF_{SCHED,ORCH}_BUFFERS_PER_THREAD` so the recycle pool has more
  headroom.
- A non-zero `current_buf_ptr` after `stop()` is logged as ERROR
  and never recovered — host treats device flush as the sole data
  path. Such a leftover indicates an AICPU flush bug, not a tail
  loss to tune around.
- `a2a3sim` exercises the export pipeline; the simulated device
  clock is not realistic for absolute-timing analysis. Use real
  hardware for steady-state numbers.

### 7.2 a5

- Buffers are fixed-size but **rotated**, so a long run is not bounded by
  one buffer's capacity. A record is dropped when the producer cannot get a
  replacement buffer — the pool's free queue is empty because the host has
  not refilled it yet — or when the ready-queue enqueue fails; AICore
  additionally drops on its own slot guard when its free queue is empty.
  Hitting `PLATFORM_PROF_BUFFER_SIZE` / `PLATFORM_PHASE_RECORDS_PER_THREAD`
  inside a buffer is a defensive path that rotation should have prevented.
  Every drop bumps `dropped_record_count`, which the host surfaces in the
  finalize log line. The tuning knobs are therefore **pool depth and
  replenishment rate** — `PLATFORM_PROF_BUFFERS_PER_CORE`,
  `PLATFORM_AICORE_BUFFERS_PER_CORE`,
  `PLATFORM_PROF_{SCHED,ORCH}_BUFFERS_PER_THREAD`, `PLATFORM_PROF_SLOT_COUNT`
  in [platform_config.h](../../src/a5/platform/include/common/platform_config.h)
  — not the per-buffer record counts.
- `a5sim` exercises the export pipeline; the simulated device
  clock is not realistic for absolute-timing analysis.

### 7.3 Common

- Chip-swimlane collection is disabled when `--rounds > 1` is in use.
- The current implementation captures incore-level scope only —
  L3 composition and orchestrator-internal sub-tasks are visible
  through the orchestrator phase summary, not as nested swimlane
  scopes.

## 8. FAQ / Debug Guide

**No `chip_swimlane_records.json` produced.** Check that
`--enable-chip-swimlane` was passed. Verify `<output_prefix>` exists
in the run log; `--rounds > 1` disables chip-swimlane collection.

**`merged_swimlane.json` is missing.** `swimlane_converter` runs
automatically after a SceneTest with `--enable-chip-swimlane`; if it
did not, run it manually:

```bash
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/chip_swimlane_records.json
```

**All tasks show as `task(rXtY)` (undistinguished).** No `deps.json`
was available, so the converter could not resolve `func_id` (it's
`-1` on disk) and every lane falls back to the anonymous `task(rXtY)`
label with no dependency arrows. Re-run with `--enable-dep-gen`, or
drop a `deps.json` from a prior `dep_gen` capture next to the records
and re-run the converter — see
[§3.5](#35-dependency-arrows-from-dep_gen).

**Tasks show as `func_<id>` instead of human names.** `deps.json`
resolved the `func_id`, but the CALLABLE spec lacks `"name"` fields
or `name_map_<case>.json` was not produced. See [profiling-name-map.md](profiling-name-map.md).

**Some tasks missing from the swimlane.** Likely dropped on device
because the buffer pool ran out. On a2a3 check
`reconcile_counters()` output for non-zero `dropped`; raise
`PLATFORM_PROF_BUFFERS_PER_CORE` /
`PLATFORM_PROF_{SCHED,ORCH}_BUFFERS_PER_THREAD`. On a5 raise
`PLATFORM_PROF_BUFFER_SIZE`.

**`current_buf_ptr` non-empty at finalize on a2a3.** The host logs
this as ERROR and does not recover. AICPU did not flush its
active chip swimlane buffer at run end. Check the AICPU flush path runs
for every thread that produced records.

**Phase records empty.** Either the runtime did not emit phase
data or phase initialization did not run. Both a2a3 runtimes and a5
`tensormap_and_ringbuffer` provide the dummy-task path; a5
`host_build_graph` does not. On both architectures collection is gated on
`ChipSwimlaneDataHeader::num_sched_phase_threads > 0` (sched) or
`num_orch_phase_threads > 0` (orch). Verify the runtime calls
`chip_swimlane_aicpu_init_phase()` in its scheduler init path; check the host's
`ChipSwimlaneCollector::initialize` zero-inits the relevant metadata
fields.

**`dispatch_time_us` < `finish_time_us` mismatch.** Verify the runtime
overwrites `task_id` with the full encoding on FIN
(`tensormap_and_ringbuffer` does
`(ring_id << 32) | local_id`); a half-filled record means AICore
wrote the WIP slot but AICPU never committed.

**Scheduler-overhead deep-dive missing from converter output.**
This is expected: the converter does not run `sched_overhead_analysis`.
Capture `deps.json` and `chip_swimlane_records.json` in separate runs, then
pass both paths to the analysis CLI; see `simpler_setup/tools/README.md`.

## 9. Related docs

- [l2-timing.md](l2-timing.md) — the everyday L2 numbers: `[STRACE]`
  host_wall / device_wall, plus Total / Orch / Sched straight from the
  `SIMPLER_DFX` device-log markers (no swimlane capture, works with
  `--rounds > 1`); the lighter alternative when you don't need the
  per-task / phase deep dive.
- [profiling-framework.md](profiling-framework.md) — shared
  host-side collector framework (a2a3 only).
- [profiling-name-map.md](profiling-name-map.md) — `func_id` →
  human name mapping for swimlane labels.
- [chip-level-arch.md](../chip-level-arch.md) — host / AICPU /
  AICore program boundaries this feature spans.
- [task-flow.md](../task-flow.md) — where AICPU dispatch and
  completion sit in the per-task state machine.
- `simpler_setup/tools/README.md` — `swimlane_converter` /
  `sched_overhead_analysis` CLI reference.
