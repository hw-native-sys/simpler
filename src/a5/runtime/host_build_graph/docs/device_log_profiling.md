# A5 HBG Device-Log Profiling

## Execution Boundary

`host_build_graph` executes orchestration on the host before device launch.
Resident AIC/AIV workers schedule and execute tasks; AICPU handles startup,
supervision, argument-dump services and retirement. Device logs therefore contain
no orchestration-thread block or device-side orchestration `dlopen` timing.
Host construction and bind timing belong in host diagnostics.

## Finding the Log

On hardware, AICPU log records are written by CANN's dlog subsystem:

```text
$HOME/ascend/log/debug/device-<device_id>/device-<pid>_<timestamp>.log
```

Filter by the current run's process and time, then inspect `A5 HBG`,
`AicpuExecutor`, `aicpu_execute`, discovery, initialization and retirement
records. Correlate the log with the native run identity and terminal result;
lines from another run are not evidence for the current launch.

## What the Records Mean

A normal launch discovers workers, validates the cluster topology, publishes
contexts, opens the execution gate, supervises the graph and retires the cores.
AICPU does not emit a per-task dispatch-loop summary for A5 HBG. Do not expect
`tasks_scheduled`, `sched_start`, or old AICPU scan/pop/setup breakdowns from
this runtime.

Scheduler task timing and phases live in the resident capture. At level 2 and
above, AICPU lifecycle records describe handshake, configuration, context
publication, bootstrap wait, gate release and exit. Per-task Completion,
Resolve, StateProbe, Dispatch and Refill belong to the AICore scheduler.
See [profiling levels](profiling_levels.md).

On failure, inspect the first error identity and error site, the timeout
classification when present, each partition's retirement result and the final
run status. A ready-pool snapshot is a diagnostic observation, not a decision
that the graph is complete. Logs must not substitute for EXIT/ACK or terminal
publication evidence.

## Startup Failures

Unknown mode or invalid bootstrap metadata is a failed launch. AICore reports
its current identity, avoids scheduler backing, and waits for EXIT/nonce ACK
and teardown release. AICPU discovers the reported cores and completes failure
retirement. This path can produce a failed terminal result without a task trace.
Missing task records must not be interpreted as a successful empty run.

## Related Tools

- `python -m simpler_setup.tools.strace_timing` summarizes host/device markers.
- Chip swimlane and the independent native protocol inspect scheduling,
  execution identities and dependency causality.
- `simpler_setup.tools.core_swimlane` reconstructs a captured task's intra-core
  execution from resident argument dumps.
- The repository's DFX workflow combines timing, dependencies and task records.
