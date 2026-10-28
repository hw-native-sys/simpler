# `host_build_graph` Device-Log Profiling

## Execution Boundary

`host_build_graph` runs its orchestration shared object on the host before the
device launch. The device boots scheduler-only, so its device log contains no
orchestrator-thread block and no device-side `dlopen` timing.

Host graph-construction timing belongs in host-side diagnostics. This guide is
only for the AICPU scheduler portion of a run.

## Finding the Log

On hardware, AICPU `LOG_INFO` records are written by CANN's dlog subsystem:

```text
$HOME/ascend/log/debug/device-<device_id>/device-<pid>_<timestamp>.log
```

Find the newest file and filter the current scheduler records:

```bash
ls -t "$HOME/ascend/log/debug/device-<device_id>"/device-*.log | head -1
grep -E "sched_start=|Scheduler summary|Scheduler Phase Breakdown" <logfile>
```

## Scheduler Summary

With `SIMPLER_DFX` enabled, each scheduler thread emits timing bounds and a
summary when its dispatch loop completes:

```text
Thread 0: sched_start=... sched_end=... sched_cost=3477.420us
Thread 0: Scheduler summary: total_time=3460.100us, loops=147, tasks_scheduled=352
```

| Field | Meaning |
| ----- | ------- |
| `sched_cost` | Wall time from scheduler-loop entry to exit |
| `total_time` | Accounted complete + dispatch + idle cycles |
| `loops` | Scheduler-loop iterations |
| `tasks_scheduled` | Mixed tasks this thread completed |

All launched AICPU threads participate in scheduling their assigned cores. The
highest-index thread performs one-time prebuilt-runtime attachment before it
enters the same scheduler path; it is not an orchestrator thread.

## Optional Phase Breakdown

When `SIMPLER_SCHED_PROFILING` is also enabled, the summary includes:

```text
Thread 0: === Scheduler Phase Breakdown: total=3460.100us, 352 tasks ===
Thread 0:   complete       : ...
Thread 0:     poll         : ... hit=... miss=... hit_rate=...%
Thread 0:   dispatch       : ...
Thread 0:     pop          : ... work=... wait=... atomics=...
Thread 0:     setup        : ...
Thread 0:   idle           : ...
Thread 0:   avg/complete   : ...
```

- `complete` polls core completion and retires mixed tasks.
- `dispatch` selects ready work, builds payloads, and publishes core commands.
- `pop` separates ready-queue work from contention wait.
- `setup` covers payload/MMIO preparation.
- `idle` is accounted time when the loop made no progress.

Dependency fanout/fanin aggregates are derived from `deps.json` by the DFX
analysis tools; current HBG device logs do not print the obsolete adjacency-list
statistics described by older guides.

## Timeout Record

A scheduler timeout uses an explicit end marker:

```text
Thread 0: sched_start=... sched_end(timeout)=... sched_cost=...
```

Use the accompanying runtime-status and stall-classification records to identify
the failure. Do not infer an orchestration stall from a missing Thread-3 block;
that block never exists in this runtime.

## Related Tools

- `python -m simpler_setup.tools.strace_timing` summarizes host/device run markers.
- `simpler_setup.tools.core_swimlane` reconstructs intra-core execution from task
  records. Explicit `enable_dump_args` requests also collect task-time argument
  metadata and selected payloads through the resident rendezvous producer;
  see [ArgsDump](../../../../../docs/dfx/args-dump.md).
- The repository's DFX analysis workflow combines scheduler timing with
  dependency and task records.
