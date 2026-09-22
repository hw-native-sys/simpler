# Device-side phase timing — fixed AICPU phases + the variable-phase plan

The AICPU **cannot write the host log on its hot path** — logging perturbs the
very timing it would measure, and the AICore side has no log path at all.
Instead it **stamps raw `get_sys_cnt_aicpu()` cycles into a host-allocated
buffer**; the host reads the buffer back after stream-sync and converts cycles
→ ns. This is how `device_wall` has always worked; this doc describes the
generalization to a fixed set of phases, and the design for variable phases.

## Fixed phases (implemented)

The on-NPU portion of `simpler_run`'s blocking wait subdivides into a **small,
closed set** of phases, each of which fires **exactly once per run per AICPU
thread**. Because the cardinality is fixed, a plain indexed store suffices — no
ring, no rotation, no recycling. This is the same model the original single
run-wall `{start, end}` pair used, generalized to `N` pairs.

`enum class AicpuPhase` (`src/common/platform/include/common/device_phase.h`):

| Phase | Window |
| ----- | ------ |
| `RunWall` | whole-run AICPU wall (the legacy `device_wall`, slot 0) |
| `Preamble` | executor init + wait-for-init before orchestration |
| `SoLoad` | orchestration-SO `dlopen` (real cost only on a first/changed callable; ~0 when cached) |
| `GraphBuild` | `orchestrate()` — submit tasks (the scheduler dispatches concurrently). **A container** for the three prep sub-phases below plus the `OrchWindow`/`SchedWindow` work. |
| `ConfigValidate` | (inside `GraphBuild`, orch thread) `config_func()` + arg-count validate |
| `ArenaWire` | (inside `GraphBuild`, orch thread) attach the prebuilt runtime arena + wire device pointers |
| `SmReset` | (inside `GraphBuild`, orch thread) SM/ring reset + finalize + bind, up to releasing the scheduler threads |
| `PostOrch` | last-thread teardown after the run completes — a2a3 deinit invalidates host-DMA/SDMA-published state before reset; a5 resets scheduler state without cache invalidation. Stamped only on the thread that finishes last, so it is the real teardown tail, strictly after `sched`. |
| `OrchWindow` | orchestrator thread's submit window (the former device-log `orch_start/end`) |
| `SchedWindow` | scheduler thread's dispatch window (the former device-log `sched_start/end`) |

`ConfigValidate` / `ArenaWire` / `SmReset` break out the per-run prep the
orchestrator thread does **inside `GraphBuild`, before the `OrchWindow` opens**
(the scheduler threads spin-wait on `runtime_init_ready_` across this whole
region). Before they existed this was an unattributed front-matter gap — on a
qwen3-14B 3.5k decode step it is ~22 ms, ~31 % of the device wall. They are
stamped only on the orchestrator thread (sched threads leave the slots unset; the
host reducer ignores them).

`OrchWindow` / `SchedWindow` replace the device-log `Orch` / `Sched` lines for
everyday use: the orchestrator and each scheduler thread store their
already-measured window into the host buffer, the host reduces across threads
(`Sched` = `min(start)..max(end)` over the scheduler threads) and emits them as
markers. The verbose per-thread `orch_start=…` / `sched_start=…` / `Scheduler
summary` (loops, tasks_scheduled) device-log lines are gated behind
`SIMPLER_ORCH_PROFILING` / `SIMPLER_SCHED_PROFILING` (default off) as an opt-in
deep-dive — see [l2-timing.md](l2-timing.md).

### Buffer layout & flow

* Host allocates one fixed buffer with layout `DevicePhaseBufferHeader |
  AicpuPhaseRecord[] | TaskTimingRecord[]`. Both record regions are
  **thread-major**. It resets the whole buffer each run and publishes its address
  into the AICPU SO via `set_platform_phase_base()` — a plain global + `extern "C"`
  setter, exactly like `set_platform_dump_base` / `set_platform_chip_swimlane_base`
  (onboard: `kernel.cpp` from `KernelArgs::device_wall_data_base`; sim: the host
  dlsym's the setter). **No C++ `thread_local`** (per
  [dynamic-linking.md](../dynamic-linking.md)); the per-thread slot is resolved
  from `platform_aicpu_affinity_thread_idx()` (a pthread-key index).
* Each surviving AICPU thread stamps its **own** slots — plain stores, no
  atomics. `kernel.cpp` stamps `RunWall`; the inner `aicpu_execute` / scheduler
  stamp `Preamble`/`SoLoad`/`GraphBuild`/`PostOrch`/`OrchWindow`/`SchedWindow`
  via `get_platform_phase_base()` + the affinity index — no change to any
  `extern "C"` signature.
* After every AICPU thread arrives at the existing completion gate, the last
  thread scans the task-timing dispatch records once and writes the header's
  `task_timing_tail_used` bit. This is outside the scheduler dispatch hot path;
  capture-disabled runs only pay a null-base check at this once-per-run point.
* Host first reads only the header + phase prefix and reduces each phase across
  threads as `max(end) - min(start)` (`DeviceRunnerBase::read_device_wall_ns`).
  It copies the larger task-timing tail only when the header says at least one
  tagged task was dispatched. It caches per-phase ns
  (`last_device_phase_ns`), and emits each non-zero phase as a `clk=dev`
  `[STRACE]` marker nested under `chip.run.runner_run.device_wall`
  (see [host-trace.md](host-trace.md)).

### Gating device-domain markers

The device markers can be turned off independently of the host spans so a
deployment can profile the two domains separately:

* **Device** (`clk=dev` markers): `SIMPLER_HOST_STRACE=1`, a visible
  `LOG_TIMING` level, and the runtime env `SIMPLER_DEVICE_STRACE_ENABLE` must all
  be enabled. The env is read once; `=0` disables the markers, while any other
  value (or unset) keeps them on (**default on**). Onboard, the same predicate
  controls buffer setup/reset, AICPU stamping, D2H readback, and marker
  emission. A disabled predicate leaves the device base null, so stamping
  no-ops and the marker-only H2D/D2H transfers are skipped.
* **Host** (`chip.run` / `bind` / `runner_run` / `validate` spans): no new
  knob — they ride the compile-time `SIMPLER_HOST_STRACE` macro and the log level
  (`LOG_TIMING`), so raising the log threshold drops them.

The simulator still applies `SIMPLER_DEVICE_STRACE_ENABLE` at emission time;
the transfer optimization above is specific to the onboard device buffer.

`RunWall` is the whole on-NPU wall (the former `RunTiming.device_wall`); it is
emitted as the `chip.run.runner_run.device_wall` marker, not returned.

### Reading the phases — they nest and overlap, don't sum them

The phases are **not a flat partition** of the wall — they nest, and some run
concurrently. Use each span's `ts` (device-domain start offset) + `dur` to see
the structure, not the bare durations:

```text
device_wall (RunWall)                 ── time flows top → bottom (sequential) ──
│
├─ preamble                              init + wait-for-init
│
├─ graph_build  (CONTAINER):
│    │
│    ├─ config_validate   ┐
│    ├─ arena_wire        │  prep (sequential, orch thread)
│    ├─ sm_reset          ┘
│    │
│    └─ Effective         ← orch & sched run SIDE BY SIDE (concurrent):
│         ┌──── orch (OrchWindow) ────┐   ┌──── sched (SchedWindow) ────┐
│         │ orchestrator submits tasks│   │ scheduler dispatches/drains │
│         └───────────────────────────┘   └─────────────────────────────┘
│              (finishes first)                 (outlasts orch)
│
└─ post_orch                             last-thread teardown (deinit), after sched
```

The vertical axis is time: `preamble → graph_build → post_orch` are **sequential**
(top to bottom), and inside `graph_build` the prep phases are sequential too. Only
`orch` and `sched` are drawn **side by side** because they are **concurrent** — both
start when the schedulers are released, and `sched` outlasts `orch` (the scheduler
keeps dispatching after the orchestrator has submitted everything). `«Effective»`
is **not a stamped marker** — `orch` and `sched` are the two real spans, and
`strace_timing` derives `Effective = max(orch_end, sched_end) −
min(orch_start, sched_start)` (their merged window).

So:

* `device_wall ≈ preamble + graph_build + post_orch` (left-to-right, sequential
  — `post_orch` is the teardown tail after `graph_build` returns).
* `graph_build ≈ prep + Effective`, where `prep = config_validate + arena_wire +
  sm_reset` (sequential) and `Effective = orch ∪ sched` (the concurrent window).
  You **cannot** add `orch + sched` — they overlap; use their union `Effective`.
* `graph_build − Effective` ≈ `prep` — the front matter, now broken out. See
  [l2-timing.md](l2-timing.md) for how the rounds-table reports `Effective`.
* `post_orch` is stamped **only on the thread that finishes last** (gated on
  `finished_`), so it captures just the `deinit` teardown. (An earlier version
  stamped it on every thread; an orchestrator thread that finished submitting
  early then sat idle waiting for the scheduler, and the cross-thread
  `max(end) − min(start)` reduction absorbed that orch-waits-for-sched overlap
  into `post_orch` — inflating it to tens of ms. It is now the real teardown.)

**Worked example** — qwen3-14B, 3.5k-context **decode** step on a2a3 (per-step
means): `device_wall` ≈ 71 ms, of which `graph_build` ≈ 71 ms (the container),
`Effective` ≈ 49 ms (`orch` ≈ 25 ms started at +22 ms, `sched` ≈ 49 ms), so the
~22 ms before the windows open — `config_validate + arena_wire + sm_reset` — is a
fixed per-decode prep cost that is now directly visible instead of only inferable
as `graph_build − Effective`.

### Sim

Sim captures the same phases as onboard. `RunWall` comes from the host
`steady_clock` (the way sim has always measured `device_wall`); the finer
subdivisions (preamble / so_load / graph_build / post_orch + orch / sched) are
stamped by the AICPU threads inside the dlopen'd runtime SO, exactly as onboard.

Sim drives those threads through the basic affinity gate, which — unlike the
onboard filter gate — does not populate `platform_aicpu_affinity_thread_idx()`.
The executor resolves the thread's index anyway (via its own fallback counter)
and publishes it with `platform_aicpu_affinity_set_thread_idx()` before any
stamping, so the per-thread phase-record slot resolves correctly inside the SO.
A phase whose measured duration rounds to 0 (e.g. preamble / graph_build on a
trivial example) is simply not emitted.

## Selective task-timing slots (implemented)

A lightweight way to measure a specific task's (or an interval's) AICPU
dispatch→finish window **without** enabling the chip swimlane — no collector
threads, no per-task AICore records, works in `SIMPLER_DFX=0`. See
[l2-timing.md](l2-timing.md) for how it relates to the swimlane's `finish_time`.

* **Opt-in is per task, by tagging.** Orchestration calls
  `CoreTaskArgs::set_task_timing_slot(id)` (`id` in `0..15`; forwarded by
  `CoreTaskArgsWithDeps`). Untagged is the default sentinel
  (`TASK_TIMING_SLOT_NONE = -1`); an out-of-range id fails through the standard
  invalid-arg path. **No env var and no compile gate** — tagging is the only
  switch. An untagged task's only added hot-path cost is one cache-hot sentinel
  compare; it never reads `get_sys_cnt_aicpu()`.
* **Transport reuses this same buffer.** The id rides the scheduler's hot
  `ChipTaskSlotState` in the `TaskAttrs` byte (bit 3 `is_timed` + bits 4-7 the
  0..15 tag), co-located with the other per-task scheduling flags. The 16 slots
  are a fixed `TaskTimingRecord[16]` **tail** appended after the `AicpuPhaseRecord`
  region in the same device buffer — same base pointer and per-run H2D reset.
  A 16-byte header at the front lets the host skip the tail D2H entirely when
  no task was tagged (saving 1536 bytes on a2a3 and 3584 bytes on a5 per run).
  The tail is a distinct record type (dispatch/finish, not start/end) reduced by
  **min(dispatch) / max(finish)**, not an `AicpuPhase`
  (those fire once per run/thread; a slot takes many block/subtask events).
* **Boundaries (match the chip swimlane).** `dispatch` = the earliest Scheduler
  publication of `DATA_MAIN_BASE` (after payload publish, immediately before the
  register write; the initial gated publication for early/speculative dispatch,
  not the doorbell release). `finish` = the latest Scheduler FIN observation
  across every required block/subtask (after the COND load + `rmb()`, before
  fanin/fanout/**deferred-completion** processing) — not the AICore kernel-end.
* **Aggregation is intentional.** For each slot the host reduces
  `min(dispatch)` / `max(finish)` across Scheduler threads. **Reusing one slot**
  for several tagged tasks yields the window from the earliest tagged dispatch to
  the latest tagged finish (**duplicate-slot** = merged window). **Distinct
  slots** keep each task's own window, so tooling recovers
  `finish(B) − dispatch(A)`. A **MIX** task's AIC/AIV0/AIV1 subtasks and an
  **SPMD** task's blocks (dispatched by different Scheduler threads) all fold
  into the one tagged slot.
* **Emission & reset.** The last AICPU thread marks the header used when it sees
  any dispatch, including a dispatched-but-incomplete task; the existing
  completeness check still decides what is emitted. Every complete slot
  (dispatch set, finish > dispatch) is
  emitted as a stable `clk=dev` span
  `chip.run.runner_run.device_wall.task_slot_<N>`, retaining start and
  duration so cross-slot intervals are recoverable; `Worker.run()` still returns
  `None`. All slots are reset every run, so a failed/short run cannot leak stale
  data; unset or incomplete slots are skipped. Slots share the phase timeline's
  `origin` when phases were stamped, and fall back to the earliest tagged
  dispatch when they were not (e.g. `host_build_graph`, whose device side stamps
  no orch/sched phases).
* **Thread index.** The fold uses the **Scheduler's own thread index** (not
  `platform_aicpu_affinity_thread_idx()`): `host_build_graph` hands its schedulers
  a local index because the sim affinity gate leaves the affinity idx `-1`, so
  resolving via affinity there would drop every write. Any distinct valid index
  per thread yields the same min/max reduction.
* **Dummy tasks.** `alloc_tensors` builds a kernel-less descriptor that never
  dispatches; its `TaskAttrs` start cleared (untagged) so a recycled ring slot
  cannot leak a stale tag.

Seams: `common/device_phase.h` (`TaskTimingRecord`, `reduce_task_timing_slots`,
buffer-layout helpers), `aicpu/device_phase_aicpu.h`
(`aicpu_task_timing_dispatch/finish(slot, thread_idx)`), the a2a3
`scheduler_dispatch.cpp` / `scheduler_completion.cpp` fold points, and the
host reset/readback/emit in the onboard + sim runners and `c_api_shared`.

## Variable phases (design, not yet implemented)

Phases entered an **unknown number of times** — per-submit, per-scheduler-loop,
arbitrary nesting — cannot use the fixed-slot buffer: there is no slot count
that bounds them, and the fixed buffer has **no rotation** (nobody recycles a
full buffer). These already have a home: the **chip swimlane orch/sched pools**
(`ChipSwimlaneAicpuOrchPhasePool` / `SchedPhasePool`), which carry a
`head + free_queue` ring that the host drains continuously. That is where
`chip_swimlane_aicpu_record_orch_phase` / `record_sched_phase` already write.

The two tracks split cleanly by **cardinality**, and that split also resolves
how each phase is tagged:

| aspect | fixed (this buffer) | variable (chip swimlane pool) |
| ------ | ------------------- | ----------------------------- |
| cardinality | once per run per thread | many per run (loops / nesting) |
| storage | indexed slot, no rotation | ring with `head + free_queue` rotation |
| tag | a stable slot index → a small closed `AicpuPhase` enum is fine | append-only → no slot needed; use a compile-time `FNV-1a` hash of the call-site name literal so adding a phase is a one-line change with no central registry |

The variable track is intentionally left as a follow-up: a `name_hash`-tagged
RAII scope writing the swimlane ring, with the host recovering hash → name from
the orch SO's `.rodata` (or a dedicated section) so no hand-maintained map is
needed.
