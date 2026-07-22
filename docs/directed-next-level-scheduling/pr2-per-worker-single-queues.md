# PR2 Plan: Route Single Tasks to Per-Worker Queues

## Objective

Replace the shared NEXT_LEVEL queue for non-group tasks with one FIFO queue per
stable NEXT_LEVEL worker ID. A single task becomes dispatchable only from the
queue owned by its submitted target worker.

PR1 is a prerequisite: every NEXT_LEVEL slot must already carry a validated
target worker ID.

## Scope

- Add a fixed, initialization-time registry of NEXT_LEVEL single-task queues.
- Route immediately-ready single tasks to their target worker queue.
- Route dependency-released single tasks to the same target worker queue.
- Dispatch each queue only to its owning worker.
- Leave NEXT_LEVEL group tasks on the existing shared NEXT_LEVEL queue.
- Leave the SUB shared queue and SUB dispatch behavior unchanged.

## Explicit Non-Goals

- Do not add or change public APIs.
- Do not implement the group ready queue yet.
- Do not add priorities, work stealing, rebinding, queue scanning, aging, or
  resource reservation.
- Do not change the DAG state machine or failure propagation.
- Do not remove group fallback helpers still needed before PR3.

## Planned Design

Introduce a small queue owner with a fixed mapping:

```text
stable NEXT_LEVEL worker ID -> ReadyQueue
```

The mapping is initialized after NEXT_LEVEL workers are registered and before
the scheduler starts. It is immutable while submissions are allowed, so queue
lookups need no structural synchronization. Each contained `ReadyQueue`
remains internally synchronized.

A shared routing helper receives a READY slot and applies this rule:

```text
NEXT_LEVEL single -> queue[target_worker_id]
NEXT_LEVEL group  -> legacy NEXT_LEVEL group-capable queue
SUB               -> existing SUB queue
```

Both submit-time readiness and completion-time dependency release must call
the same helper. Duplicating this routing logic is not allowed.

## Planned File Changes

- `src/common/hierarchical/types.h` and `types.cpp`
  - Add the fixed per-worker queue owner or equivalent focused abstraction.
- `src/common/hierarchical/worker_manager.h` and `worker_manager.cpp`
  - Expose the registered NEXT_LEVEL worker IDs needed for initialization.
- `src/common/hierarchical/worker.h` and `worker.cpp`
  - Own and initialize the per-worker queues.
  - Pass them to Orchestrator and Scheduler.
- `src/common/hierarchical/orchestrator.h` and `orchestrator.cpp`
  - Route newly READY single slots through the shared routing abstraction.
- `src/common/hierarchical/scheduler.h` and `scheduler.cpp`
  - Route newly released single slots identically.
  - Drain each per-worker queue only when its worker is idle.
- `tests/ut/cpp/hierarchical/test_orchestrator.cpp`
  - Verify submit-time routing by exact worker ID.
- `tests/ut/cpp/hierarchical/test_scheduler.cpp`
  - Verify dependency-release routing and independent worker progress.

## Validation

- A single task targeted to worker A never dispatches on worker B.
- A busy worker A does not block a READY single task for idle worker B.
- A PENDING task enters the correct queue when its final producer completes.
- A failed producer does not enqueue its consumer.
- Existing group and SUB tests retain their pre-PR2 behavior.

## Size Budget

- Production code: at most 450 added lines.
- Tests: at most 350 added lines.
- Documentation adjustments: at most 100 added lines.
- Total target: at most 900 added lines.

## Completion Criteria

- All NEXT_LEVEL single dispatch is exact-worker and per-worker FIFO.
- Group and SUB behavior remains unchanged.
- Submit-time and completion-time routing share one implementation.
- The PR builds and focused hierarchical C++ tests pass independently.
