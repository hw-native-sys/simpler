# L2 Timing — host_wall / device_wall / Effective / Orch / Sched (`[STRACE]`)

For an L2 run you usually look at a handful of timing numbers. They **all** come
from **`[STRACE]` markers** (gated on `SIMPLER_PROFILING`, default on in
`src/common/task_interface/profiling_config.h`): the platform emits one marker
line per stage to stderr, and `simpler_setup.tools.strace_timing` parses them
offline. Normal execution only *emits*; parsing is a separate, opt-in step.

A richer channel — the **L2 swimlane** per-task / scheduler-phase capture — is
opt-in (`--enable-l2-swimlane`) and documented separately in
[l2-swimlane-profiling.md](l2-swimlane-profiling.md).

> Don't confuse any of these with the host-side runtime-stage seconds a test
> harness prints (`[RUN] runtime done (Ns)`): that is wall-clock around JIT
> compile + CPU golden + Python dispatch, ~1000× the on-device µs below.

## 1. The markers — host_wall / device_wall

`simpler_run()` emits one `[STRACE]` log line per stage (see
[host-trace.md](host-trace.md)). The two headline walls:

| Span | What it measures | Source |
| ---- | ---------------- | ------ |
| **`simpler_run`** (host_wall) | Host `steady_clock` delta wrapping the dispatch call — includes Python/host overhead. | host side, around the C-ABI run call |
| **`simpler_run.runner_run.device_wall`** | **Full on-NPU kernel wall**: earliest `simpler_aicpu_exec` start to latest end across launched threads — i.e. **the whole run + teardown**. | Each `simpler_aicpu_exec` thread stamps its own `AicpuPhase::RunWall` slot in the per-thread `AicpuPhaseRecord` buffer (`KernelArgs.device_wall_data_base`, see `src/{arch}/platform/onboard/aicpu/kernel.cpp`); host reduces `max(end) - min(start)` each run |

Both are emitted whenever the runtime was built with `SIMPLER_PROFILING` (the
default) — **independent of `--enable-l2-swimlane`**. The `device_wall` marker is
absent only on a `SIMPLER_PROFILING`-off build.

Nested under `device_wall` are the AICPU orchestrator / scheduler sub-spans
(`clk=dev`; captured on both onboard and sim):

| Span | Window |
| ---- | ------ |
| **`…device_wall.orch`** (Orch) | orchestrator run window — graph construction on the AICPU. |
| **`…device_wall.sched`** (Sched) | scheduler dispatch/execution window. |

**`device_wall` is the full kernel wall, NOT the orchestration span** — it is
strictly larger than Orch/Sched/Effective below, because it also covers AICPU
init and exec teardown around the orchestrate+schedule work.

For a finer per-stage breakdown of `device_wall` (preamble / SO-load /
graph-build / post-orch) and of the host side (`bind` / `runner_run` /
`validate`), see [host-trace.md](host-trace.md) and the device-phase mechanism in
[device-phases.md](device-phases.md) — both ride on the same `SIMPLER_PROFILING`
macro, no extra flag.

## 2. Per-round table — `strace_timing --rounds-table`

To get a per-round table for a `--rounds N` run, **tee stderr to a file and parse
it offline**:

```bash
python test_*.py -p a2a3 -d 0 --rounds 100 --skip-golden > run.log 2>&1
python -m simpler_setup.tools.strace_timing run.log --rounds-table
```

```text
  Round     Host (us)   Device (us)  Effective (us)     Orch (us)    Sched (us)
  -----------------------------------------------------------------------------
  0          470000.0        9050.0          931.0         540.0         930.0
  ...
  Avg Host: 468000.0 us  |  Avg Device: 9010.0 us [100/100]  |  Avg Effective: 930.0 us  |  Avg Orch: 539.0 us  |  Avg Sched: 928.0 us  (100 rounds)
```

One column per `[STRACE]` wall found — **Host** always, plus **Device** /
**Orch** / **Sched** / **Effective** whenever those markers were captured
(onboard and sim both capture the device-domain subdivision). A column is hidden
when every round read 0, and an individual phase whose duration rounds to 0 is
not emitted. Because the offline tool groups markers by `(pid, inv)`, this works
for **L3 multi-round too** (each chip-child's `simpler_run` is its own
invocation).

### Column meanings

| Column | Definition |
| ------ | ---------- |
| **Host** | `simpler_run` span — host wall incl. Python/dispatch overhead. |
| **Device** | `device_wall` span — full on-NPU wall incl. init/teardown. |
| **Effective** | `max(orch_end, sched_end) − min(orch_start, sched_start)` — the orch∪sched merged window (the effective on-device execution window). Computed from the orch/sched markers' **device-domain** `ts`+`dur` (the device spans carry a device-clock start offset, not the host emit time). This is the old device-log "Total", now derived purely from the markers — no device log needed. |
| **Orch** | `…device_wall.orch` span — orchestrator (graph-build) window. |
| **Sched** | `…device_wall.sched` span — scheduler dispatch/execution window. |

```text
device_wall  =  init  +  [ orchestrate + schedule/execute ≈ Effective ]  +  teardown
                 ^^^^                                                       ^^^^^^^^
                 only in device_wall, not in Effective/Orch/Sched
```

Use Orch/Sched/Effective to see *where inside the run* time went (graph build vs
scheduling); use `device_wall` for the end-to-end on-NPU cost including
init/teardown.

`tools/benchmark_rounds.sh` runs this tool for you across a set of examples (it
tees each run and renders the table, then builds a cross-example summary).

## 3. Deep dive — per-thread loops / tasks (device log, by eye)

The `[STRACE]` markers carry *durations*, not the per-thread `loops` /
`tasks_scheduled` counters. Those live only in the CANN device log, behind the
opt-in `PTO2_ORCH_PROFILING` / `PTO2_SCHED_PROFILING` macros (default **off**).
When you need them, rebuild with the macro on and **read the device log
directly** — there is no longer a Python parser for it:

```bash
# build the runtime with the deep-dive lines compiled in
CXX="g++ -DPTO2_SCHED_PROFILING=1" python simpler_setup/build_runtimes.py --platforms a2a3
# run, then grep the device log by eye
grep -E 'orch_start=|sched_start=|Scheduler summary' \
    ~/ascend/log/debug/device-0/device-*.log
```

```text
Thread 3: orch_start=86702071725377 orch_end=86702071752336 orch_cost=539.180us
Thread 0: sched_start=86702071724963 sched_end=86702071771552 sched_cost=931.780us
Thread 0: Scheduler summary: total_time=923.660us, loops=743, tasks_scheduled=181
```

For a per-task / Head-Tail / scheduler-phase breakdown, use the swimlane capture
instead (see Related docs).

## 4. Limitations

- **Per-thread `loops` / `tasks_scheduled` counters** are not in the markers —
  for those, rebuild with `PTO2_SCHED_PROFILING=1` and read the device log (§3).
- **`Worker.run` returns `None`** — timing is never a return value; it is only
  ever read from the markers.

## 5. Related docs

- [host-trace.md](host-trace.md) — the `[STRACE]` marker grammar and the host /
  device per-stage spans.
- [device-phases.md](device-phases.md) — how the AICPU phase windows
  (`RunWall` / orch / sched / …) are stamped and read back.
- [l2-swimlane-profiling.md](l2-swimlane-profiling.md) — the per-task /
  scheduler-phase deep dive.
- `simpler_setup/tools/README.md` — `strace_timing` CLI reference.
