# A5 HBG Profiling Levels

## Build-Time Controls

Defaults and dependency checks are centralized in
`src/common/task_interface/profiling_config.h`. Runtime and kernel producers and
consumers must be built against the same current definitions.

| Macro | Default | Scope |
| ----- | ------- | ----- |
| `SIMPLER_DFX` | 1 | Compile diagnostic collectors and associated recording paths |
| `SIMPLER_HOST_STRACE` | 1 | Host `[STRACE]` markers and device-domain marker capture; independent of DFX collectors |
| `SIMPLER_ORCH_PROFILING` | 0 | Host orchestration timing; requires DFX |
| `SIMPLER_SCHED_PROFILING` | 0 | Additional scheduler instrumentation where used; requires DFX |
| `SIMPLER_TENSORMAP_PROFILING` | 0 | Host TensorMap instrumentation; requires orchestration profiling |

These are build controls. The per-run chip-swimlane level controls which
compiled collectors publish data. Neither a compile-time flag nor a capture
level removes required error checks, dependency ordering or retirement gates.
Do not add a scheduler choice to a diagnostic flag.

## Per-run data

The [Chip Swimlane guide](../../../../../docs/dfx/chip-swimlane-profiling.md#a5-hbg-calendar)
documents L0 through L4. The calendar's AICore Worker View shows the kernel
interval per physical lane. Scheduler View shows dependency, launch and
completion transitions, including deferred completion. The AICPU view
shows range materialization and supervision. PGO training uses separate timing
slots selected by the Host.

Each execution is identified by run epoch, task ID, block, and role. A Graph
child's task ID names the invocation as well as the child. An argument capture
records inputs before dispatch and selected outputs after the last participant
finishes. PMU samples likewise refer to the kernel's actual task and lane.

Use [device-log profiling](device_log_profiling.md) for launch and completion
errors and the [runtime design](RUNTIME_LOGIC.md) for storage and retirement.
