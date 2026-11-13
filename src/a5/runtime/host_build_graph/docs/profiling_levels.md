# A5 HBG Profiling Levels

A5 `host_build_graph` builds its graph on the host and schedules tasks on
resident AIC/AIV workers. AICPU performs discovery, configuration, supervision
and retirement. Diagnostic flags do not select an execution path: ordinary DAGs
and native Graph replay both use resident scheduling.

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

## Per-Run Capture Levels

The host snapshots `chip_swimlane_level` in the run descriptor and resident run
control. The platform entry carries the binary activation flag; AICore reads the
run-control level after READY. The capture's document-level `metadata.runtime`
identifies the runtime, and scheduler records carry their producer identity.

| Level | Resident HBG data |
| ----- | ----------------- |
| 0 | Capture disabled |
| 1, `TASK_TIMING` | AICore task execution timing |
| 2, `SCHEDULE_TIMING` | Per-task scheduling timestamps and AICPU lifecycle timing |
| 3, `SCHED_PHASES` | Scheduler phases and taskless-loop intervals |
| 4, `ORCH_PHASES` | Host orchestration/bind detail and resident work-protocol data |

A bare `--enable-chip-swimlane` uses level 4. Disabling DFX at build time makes
its associated collection paths unavailable regardless of the requested level.
Host timing is not an AICPU orchestration-thread measurement: this runtime has
no device orchestrator.

At level 2 and above, lifecycle records contain per-AICPU-thread handshake,
configuration, context publication, bootstrap wait, execution-gate release and
exit intervals. Configuration is leader-owned. These records describe the
control plane and must not be interpreted as per-task AICPU dispatch.

## Task and Scheduler Identity

Kernel execution identity is `(run_epoch, node_ref, block_idx, subtask_idx)`.
The public task identity and native Graph `(original_shell_id, local_child_id)`
remain separately available for attribution. ENTER/EXIT/control nodes have no
kernel execution record. The producer must not fabricate an AICore interval for
one of these nodes.

Resident scheduling captures identity explicitly from the selected node and
physical execution. `LocalContext.async_ctx.task_token` is not the source of
resident task identity. Argument slots 48/49 still address the current local and
global kernel contexts; their presence does not imply AICPU task dispatch.

`dispatch_time` marks dispatch publication completion; `finish_time` marks the
scheduler's processing of the returned execution. Phase records distinguish
Completion, Resolve, StateProbe, Dispatch and Refill. A directly retained
successor does not invent a ready-pool probe. A ready-pool replacement retains
its actual probe even when it reuses a completed local slot.

At level 3, consecutive taskless iterations may coalesce into `idle` intervals.
Fixed-capacity trace buffers do not wrap silently: dropped/truncated indicators
must accompany incomplete captures. Neither an idle interval nor an empty ready
queue proves graph completion.

The resident executor owns per-task trace staging/publication and AICore buffer
rotation. AICPU settles run diagnostics only after retirement is proved. The
AICPU-dispatched completion-before-next-dispatch rotation model used by other
runtimes is not the A5 HBG recording contract.

## Host Bind and Protocol Sources

Host phase records distinguish orchestration, Definition packing, image staging
and actual metadata publication. Bind and device execution have separate timing
boundaries; a reduction in bind work is not a measured Device latency reduction.
See the repository's `hbg-bind-phases` and `dfx-analyze` workflows.

Native Graph protocol validation has two inputs: the executed native image and
an independent snapshot of original outer tasks plus recorded Definitions.
The Host `original_definition_section` preserves that source snapshot. It is
not a device launch-envelope carrier and must not be replaced by expanded
bindings from the planner being checked.

## Errors, Dumps and Verification

Argument dumps are produced from resident execution boundaries and copied by
AICPU dump services. `core_swimlane` replays one captured physical task; use chip
swimlane and protocol inspection for cross-task or dependency questions.

A startup failure may have no usable scheduler backing or task trace. Its
observable result is a failed launch status plus the retirement outcome. A failed
or incomplete capture does not turn that run into a successful empty graph.
Retained dump/trace owners may be settled or reused only under their actual
retirement and open-record contracts.

Use [device-log profiling](device_log_profiling.md) for cold-path diagnostics,
[the runtime design](RUNTIME_LOGIC.md) for lifecycle ordering, and
[the Graph protocol](../../../../../docs/zh-cn/design/a5-hbg-graph-task/protocol.md)
for node identity and completion semantics. Validation reports must identify the
exact runtime build, capture level, truncation state and checks actually run.
