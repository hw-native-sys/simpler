# PR3 Plan: Add Directed Group Dispatch

## Objective

Move NEXT_LEVEL group tasks to a dedicated FIFO queue and dispatch a group only
when all user-selected workers are idle. Claim and dispatch the complete worker
set as one scheduler action.

PR1 and PR2 are prerequisites: group members carry validated exact targets,
and single tasks already use per-worker queues.

## Scope

- Give NEXT_LEVEL groups a dedicated FIFO ready queue.
- Route immediately-ready and dependency-released groups to that queue.
- Inspect only the queue head.
- Resolve every member to its submitted stable worker ID.
- Dispatch only when all target workers are idle.
- Dispatch a launchable group before per-worker single queues in the same
  scheduler iteration.
- If the head group cannot launch, leave it READY and continue with single
  queues without reserving any worker.
- Preserve current group completion aggregation and failure poisoning.

## Explicit Non-Goals

- Do not scan past the group queue head.
- Do not reserve a subset of group workers.
- Do not add fairness, aging, priorities, quotas, preemption, or starvation
  prevention.
- Do not change SUB group scheduling.
- Do not change group dependency semantics: one group remains one DAG node.

## Planned Dispatch Rule

```text
group = group_ready_queue.front
if group exists and every target worker is idle:
    pop group
    mark slot RUNNING
    dispatch every member to its exact target

for each NEXT_LEVEL worker:
    if worker is idle:
        dispatch the head of that worker's single queue
```

`WorkerThread::dispatch` marks a worker non-idle synchronously. The scheduler
is the only dispatch producer, so resolving all workers before the first
dispatch prevents partial launch caused by an ordinary busy-worker check.

## Planned File Changes

- `src/common/hierarchical/types.h` and `types.cpp`
  - Add the dedicated group queue to the directed queue owner.
- `src/common/hierarchical/orchestrator.cpp`
  - Route READY NEXT_LEVEL groups to the group queue.
- `src/common/hierarchical/scheduler.cpp`
  - Add launchable-head group dispatch before single dispatch.
  - Remove idle-pool filling for NEXT_LEVEL groups.
  - Keep SUB dispatch on its existing path.
- `tests/ut/cpp/hierarchical/test_scheduler.cpp`
  - Verify exact member-to-worker mapping.
  - Verify no member dispatches while one target is busy.
  - Verify a launchable group wins over conflicting queued singles.
  - Verify a blocked group does not reserve idle workers.
  - Verify group completion releases downstream consumers only once.

## Failure and Concurrency Checks

- Duplicate target IDs are rejected by PR1, before queue insertion.
- Worker lookup failure is treated as an invariant violation, not as a request
  for fallback selection.
- Existing group member failure handling remains responsible for terminal
  aggregation and downstream poison.
- No new mutex may be held while an endpoint executes.

## Size Budget

- Production code: at most 350 added lines.
- Tests: at most 400 added lines.
- Documentation adjustments: at most 100 added lines.
- Total target: at most 850 added lines.

## Completion Criteria

- NEXT_LEVEL groups use only their exact submitted workers.
- Group allocation is all-or-nothing at dispatch time.
- The only cross-queue policy is launchable-group-first.
- No reservation, scan, rebinding, or fairness mechanism is introduced.
- Focused group, dependency, and failure tests pass independently.
