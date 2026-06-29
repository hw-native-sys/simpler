# Device-side phase timing — fixed AICPU phases + the variable-phase plan

The AICPU **cannot write the host log on its hot path** — logging perturbs the
very timing it would measure, and the AICore side has no log path at all.
Instead it **stamps raw `get_sys_cnt_aicpu()` cycles into a host-allocated
buffer**; the host reads the buffer back after stream-sync and converts cycles
→ ns. This is how `device_wall` has always worked; this doc describes the
generalization to a fixed set of phases, and the design for variable phases.

## Fixed phases (implemented)

The on-NPU portion of `run_prepared`'s blocking wait subdivides into a **small,
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
| `GraphBuild` | `orchestrate()` — submit tasks (the scheduler dispatches concurrently) |
| `PostOrch` | runtime-status read + last-thread cleanup after orchestration |
| `OrchWindow` | orchestrator thread's submit window (the former device-log `orch_start/end`) |
| `SchedWindow` | scheduler thread's dispatch window (the former device-log `sched_start/end`) |

`OrchWindow` / `SchedWindow` replace the device-log `Orch` / `Sched` lines for
everyday use: the orchestrator and each scheduler thread store their
already-measured window into the host buffer, the host reduces across threads
(`Sched` = `min(start)..max(end)` over the scheduler threads) and emits them as
markers. The verbose per-thread `orch_start=…` / `sched_start=…` / `Scheduler
summary` (loops, tasks_scheduled) device-log lines are gated behind
`PTO2_ORCH_PROFILING` / `PTO2_SCHED_PROFILING` (default off) as an opt-in
deep-dive — see [l2-timing.md](l2-timing.md).

### Buffer layout & flow

* Host allocates one `AicpuPhaseRecord[NUM_AICPU_PHASES]` per launched AICPU
  thread (**thread-major**), resets it each run, and publishes its address into
  the AICPU SO via `set_platform_phase_base()` — a plain global + `extern "C"`
  setter, exactly like `set_platform_dump_base` / `set_platform_l2_swimlane_base`
  (onboard: `kernel.cpp` from `KernelArgs::device_wall_data_base`; sim: the host
  dlsym's the setter). **No C++ `thread_local`** (per
  [dynamic-linking.md](../dynamic-linking.md)); the per-thread slot is resolved
  from `platform_aicpu_affinity_thread_idx()` (a pthread-key index).
* Each surviving AICPU thread stamps its **own** slots — plain stores, no
  atomics. `kernel.cpp` stamps `RunWall`; the inner `aicpu_execute` / scheduler
  stamp `Preamble`/`SoLoad`/`GraphBuild`/`PostOrch`/`OrchWindow`/`SchedWindow`
  via `get_platform_phase_base()` + the affinity index — no change to any
  `extern "C"` signature.
* Host reduces each phase across threads as `max(end) - min(start)` on readback
  (`DeviceRunnerBase::read_device_wall_ns`), caches per-phase ns
  (`last_device_phase_ns`), and emits each non-zero phase as a `clk=dev`
  `[STRACE]` marker nested under `run_prepared.runner_run.device_wall`
  (see [host-trace.md](host-trace.md)).

`RunWall` is the whole on-NPU wall (the former `RunTiming.device_wall`); it is
emitted as the `run_prepared.runner_run.device_wall` marker, not returned.

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

## Variable phases (design, not yet implemented)

Phases entered an **unknown number of times** — per-submit, per-scheduler-loop,
arbitrary nesting — cannot use the fixed-slot buffer: there is no slot count
that bounds them, and the fixed buffer has **no rotation** (nobody recycles a
full buffer). These already have a home: the **L2 swimlane orch/sched pools**
(`L2SwimlaneAicpuOrchPhasePool` / `SchedPhasePool`), which carry a
`head + free_queue` ring that the host drains continuously. That is where
`l2_swimlane_aicpu_record_orch_phase` / `record_sched_phase` already write.

The two tracks split cleanly by **cardinality**, and that split also resolves
how each phase is tagged:

| aspect | fixed (this buffer) | variable (L2 swimlane pool) |
| ------ | ------------------- | --------------------------- |
| cardinality | once per run per thread | many per run (loops / nesting) |
| storage | indexed slot, no rotation | ring with `head + free_queue` rotation |
| tag | a stable slot index → a small closed `AicpuPhase` enum is fine | append-only → no slot needed; use a compile-time `FNV-1a` hash of the call-site name literal so adding a phase is a one-line change with no central registry |

The variable track is intentionally left as a follow-up: a `name_hash`-tagged
RAII scope writing the swimlane ring, with the host recovering hash → name from
the orch SO's `.rodata` (or a dedicated section) so no hand-maintained map is
needed.

## Aligning host and device on one timeline (future)

Host spans use `CLOCK_MONOTONIC` ns; device phases use AICPU cycles. They are
reported in the same `[STRACE]` grammar (`clk=dev` marks the cycle-derived
durations). To place both on **one** Perfetto timeline later, the device side
would emit a periodic `clock.anchor` line pairing one `host_ns` with one
`dev_cyc` + `dev_freq`; the parser then maps device spans onto the host axis.
The marker `v=` field and `k=v` extensibility reserve room for this — it is not
part of the current host-only work.
