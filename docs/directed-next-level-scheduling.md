# Directed NEXT_LEVEL Scheduling

Every NEXT_LEVEL task names its exact worker at submission time, and the
Scheduler dispatches it only to that worker. The runtime never selects a
NEXT_LEVEL worker on the caller's behalf. SUB tasks keep free scheduling and
take no worker id.

This page is the placement contract. For the queue topology, dispatch loop,
and dependency/failure handling see [scheduler.md](scheduler.md); for the
submit API see [orchestrator.md](orchestrator.md).

## Placement contract

- `Orchestrator.submit_next_level(callable, args, config, worker=id)` targets
  exactly one stable NEXT_LEVEL worker id.
- `Orchestrator.submit_next_level_group(callable, args_list, config,
  workers=[...])` takes one stable worker id per member. The ids must be
  distinct and their count must equal the member count.
- A worker id must be non-negative and name a registered NEXT_LEVEL worker.
  Booleans, floats, numeric strings, and other coercible non-integers are
  rejected.
- A local (`LOCAL_PYTHON` / `LOCAL_CHIP`) callable must target a local child;
  a remote worker id is rejected because a remote endpoint installs only its
  own dispatcher callables.
- When a callable or its tensor data constrains eligibility (remote callables,
  remote-buffer data), the target must lie in that eligible set. Eligibility
  bounds which targets are *valid*; it is not a scheduler selection policy.
- `submit_sub` / `submit_sub_group` are unchanged and take no worker id.

## Queue and dispatch model

The worker owns a directed-queue registry fixed at `init()` from the stable
NEXT_LEVEL worker ids:

| Task | Queue |
| ---- | ----- |
| NEXT_LEVEL single | `FIFO[target_worker_id]` |
| NEXT_LEVEL group | one shared group FIFO |
| SUB | one shared SUB FIFO |

A task's target and queue never change after submission. Tasks that are READY
at submission and tasks released by a completing dependency use the same
routing operation.

Each Scheduler iteration:

- launches the group FIFO head only when every one of its target workers is
  idle — all members or none, with no partial reservation;
- dispatches the head of each idle worker's single FIFO;
- dispatches READY SUB work through the existing free-selection path.

Only the group FIFO head is examined, and a launchable group is tried before
conflicting singles. The runtime adds no fairness, aging, priority, or
reservation policy; callers choose worker sets that make acceptable progress.

## Invariants

- PENDING tasks live in task-slot state, not in a ready queue; a task is routed
  only once all producers complete successfully.
- A failed producer poisons its dependents instead of enqueuing them.
- A group is one DAG node and completes only after every member reaches a
  terminal state.
- The NEXT_LEVEL worker set is fixed after `init()`; a worker lookup that fails
  after submit-time validation is an invariant violation, not a fallback.
- No queue or scheduler mutex is held while an endpoint executes.

## Non-goals

- No SUB worker selection.
- No priorities, work stealing, rebinding, queue scanning, aging, quotas,
  preemption, starvation prevention, or partial group reservation.
- No compatibility flags, environment variables, or macros.

## Related documents

- [scheduler.md](scheduler.md) — queue topology, dispatch loop, completion
- [orchestrator.md](orchestrator.md) — submit API and DAG construction
- [remote-l3-worker-design.md](remote-l3-worker-design.md) — remote NEXT_LEVEL
  workers
