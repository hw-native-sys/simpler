# L2 Timing — host_wall / device_wall (RunTiming) + device-log Orch/Sched/Total

For an L2 run you usually look at a handful of timing numbers. They come from
**two channels**, both available with no extra flags because they ride on the
compile-time `PTO2_PROFILING` macro (default `1`, already in the prebuilt
runtimes):

1. **`RunTiming`** — `host_wall` and `device_wall`, returned directly by
   `Worker.run()` and printed per round by the harness.
2. **Device-log markers** — `Orch`, `Sched`, `Total`, parsed from the CANN
   device log (this is the finer per-thread breakdown).

A third, richer channel — the **L2 swimlane** per-task / scheduler-phase
capture — is opt-in (`--enable-l2-swimlane`) and documented separately in
[l2-swimlane-profiling.md](l2-swimlane-profiling.md).

> Don't confuse any of these with the host-side runtime-stage seconds a test
> harness prints (`[RUN] runtime done (Ns)`): that is wall-clock around JIT
> compile + CPU golden + Python dispatch, ~1000× the on-device µs below.

## 1. `RunTiming` — host_wall / device_wall

`Worker.run()` / `run_prepared()` return a `RunTiming`
(`_task_interface.RunTiming`, see `src/common/hierarchical/types.h`):

| Field | What it measures | Source |
| ----- | ---------------- | ------ |
| **`host_wall_us`** | Host `steady_clock` delta wrapping the dispatch call — includes Python/host overhead. | host side, around the C-ABI run call |
| **`device_wall_us`** | **Full on-NPU kernel wall**: from `simpler_aicpu_init` (start, captured single-threaded) to the end of `simpler_aicpu_exec` — i.e. **init + the whole run + teardown**. | AICPU stamps it into `KernelArgs.device_wall_*` (`src/{arch}/platform/onboard/aicpu/kernel.cpp`); host reads it back each run |

Both are populated whenever the runtime was built with `PTO2_PROFILING` (the
default) — **independent of `--enable-l2-swimlane`**. `device_wall_us` is `0`
only on a `PTO2_PROFILING`-off build.

**`device_wall` is the full kernel wall, NOT the orchestration span.** It is
strictly larger than the device-log `Orch`/`Sched`/`Total` below, because it
also covers AICPU init and exec teardown around the orchestrate+schedule work.

The standalone runner / pytest harness prints these per round for `--rounds N`
(`scene_test._log_round_timings`):

```text
  Round     Host (us)   Device (us)
  -------   ---------   -----------
  0          470000.0        9050.0
  ...
  Avg Host: 468000.0 us  |  Avg Device: 9010.0 us  (100 rounds)
```

## 2. Device-log markers — Orch / Sched / Total

Every on-device run emits a block of orchestrator + scheduler `LOG_INFO_V9`
lines to the CANN device log (from `aicpu_executor.cpp` and
`scheduler_cold_path.cpp`):

```text
Thread 3: orch_start=86702071725377 orch_end=86702071752336 orch_cost=539.180us
Thread 0: sched_start=86702071724963 sched_end=86702071771552 sched_cost=931.780us
Thread 0: Scheduler summary: total_time=923.660us, loops=743, tasks_scheduled=181
PTO2 total submitted tasks = 355
```

Grouped into one round per orchestrator block, reported as:

| Metric | Definition |
| ------ | ---------- |
| **Orch** | `orch_start → orch_end` on the orchestrator thread (== `orch_cost`) — graph construction on the AICPU. |
| **Sched** | `min(sched_start) → max(sched_end)` across scheduler threads — the dispatch/execution window. |
| **Total** | `min(start) → max(end)` across orchestrator + scheduler threads. |

Cycle counts are converted to microseconds with a frequency **self-calibrated
from each round's `orch_cost`** (`cycles / cost_us`), so no per-platform clock
constant is needed (`benchmark_rounds.sh` hard-codes the same ≈50 MHz for
a2a3).

These are the same markers `sched_overhead_analysis`'s device-log fallback used
before #787. They were restored as a focused capability since dropping that
fallback left no way to read timing from a plain run.

### How the three numbers relate to `device_wall`

```text
device_wall  =  init  +  [ orchestrate + schedule/execute = ~Total ]  +  teardown
                 ^^^^                                                    ^^^^^^^^
                 only in device_wall, not in the device-log Orch/Sched/Total
```

Use the device-log Orch/Sched/Total to see *where inside the run* time went
(graph build vs scheduling); use `device_wall` for the end-to-end on-NPU cost
including init/teardown.

## 3. How to read the device-log markers

### 3.1 From the test harness — `--enable-device-log-timing`

Onboard L2 only. Prints a per-round Orch/Sched/Total table after the run; works
with `--rounds N` (unlike `--enable-l2-swimlane`, disabled for multi-round).

```bash
python examples/.../test_my_example.py -p a2a3 --device 0 \
    --enable-device-log-timing --rounds 50
pytest tests/st/... --platform a2a3 --enable-device-log-timing
```

The harness snapshots the device-log directory before the run, waits out CANN's
asynchronous log flush, then parses only this run's content (see §5).

### 3.2 Standalone — `device_log_timing`

For any device log, including one produced by an external workload that never
emits an `l2_swimlane_records.json`:

```bash
# Explicit file or glob (quote the glob so the shell doesn't expand it)
python -m simpler_setup.tools.device_log_timing \
    --device-log '~/ascend/log/debug/device-0/device-*.log'

# Auto-pick the newest log under device-<id>/
python -m simpler_setup.tools.device_log_timing -d 0
```

```text
Device-log timing (from .../device-0/device-2244428_20260609115418256.log)
  tasks = 355
  Round     Total (us)     Orch (us)    Sched (us)
  ------------------------------------------------
  0              932.0         539.2         932.0
  Avg  Total: 932.0 us  |  Orch: 539.2 us  |  Sched: 932.0 us  (1 rounds)
```

With more than one round the table gets one row per round, an average, and a
trimmed average (dropping the 10 lowest + 10 highest) once there are >20 rounds.

## 4. Environment Variables

The device-log reader follows CANN's own log-location and output rules:

| Env var | Effect |
| ------- | ------ |
| `ASCEND_PROCESS_LOG_PATH` | Relocates the log root to `$ASCEND_PROCESS_LOG_PATH/debug` (highest precedence, above `ASCEND_WORK_PATH/log/debug`, above the `<euid-home>/ascend/log/debug` default). Resolved automatically. |
| `ASCEND_WORK_PATH` | Log root `$ASCEND_WORK_PATH/log/debug` when `ASCEND_PROCESS_LOG_PATH` is unset. |
| `ASCEND_SLOG_PRINT_TO_STDOUT=1` | Routes CANN logs to stdout — **no device log file is written**, so there is nothing to parse. The CLI errors out and the harness flag skips; unset it (or set `0`) to capture device timing. |
| `ASCEND_HOST_LOG_FILE_NUM` | Rotated files retained per process (default 10). Each file caps at 20 MB; a long run can rotate mid-run (see §5). |

> **Default root uses the effective uid's real home, not `$HOME`.** When no
> relocation env var is set, the root is `<euid-home>/ascend/log/debug`, where
> `<euid-home>` is the passwd home of the effective uid (e.g. `/root` for a run
> launched as root via sudo / `task-submit`). This matches where the driver
> actually writes device logs; `$HOME` is unreliable because sudo commonly
> leaves it pointing at the invoking user, so the log would never be found.
> Set `ASCEND_PROCESS_LOG_PATH` to a readable directory to override.

## 5. Multi-Round & Rotation Handling (device-log path)

1. **CANN flushes the device log asynchronously.** Immediately after a run the
   timing lines may not be on disk yet. The harness polls (up to ~20 s) until
   the fresh log carries at least the expected number of round blocks.
2. **A log file is shared and rotates.** A long run can split its blocks across
   several 20 MB files, and a reused worker process appends later cases' blocks
   to the same file. The harness therefore reads **all** files modified after
   the run started (mtime order) and snapshots each pre-existing file's byte
   size before the run, parsing only the bytes appended afterward. An explicit
   `--device-log` glob likewise parses every matched file.

A file that rotates away between discovery and read is skipped rather than
aborting the parse.

## 6. Limitations (device-log path)

- **Onboard only.** Sim platforms have no CANN device log; the harness flag is
  a no-op there. (`RunTiming` host/device wall *are* available on sim.)
- **L2 only.** An L3 run spans multiple chip processes, each with its own
  device log, so a single Total/Orch/Sched table would be ambiguous.
- **Round grouping assumes the orchestrator block precedes that round's
  scheduler lines** (true for the normal emit order).
- **Aggregate only.** No per-task / Head-Tail OH / scheduler-phase breakdown —
  use [l2-swimlane-profiling.md](l2-swimlane-profiling.md) for those.

## 7. Related docs

- [l2-swimlane-profiling.md](l2-swimlane-profiling.md) — the per-task /
  scheduler-phase deep dive; the richer alternative when a swimlane capture is
  available.
- [logging.md](../logging.md) — the `LOG_INFO_V9` device-log verbosity tier
  these markers are emitted at.
- `simpler_setup/tools/README.md` — `device_log_timing` and
  `sched_overhead_analysis` CLI reference.
