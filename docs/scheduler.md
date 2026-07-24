# Scheduler — DAG Dispatch Internals

The Scheduler is the single-threaded DAG executor for one hierarchical
`Worker`. The Orchestrator constructs slots and dependencies; the Scheduler
dispatches READY slots, processes worker completions, releases downstream
dependencies, and retires terminal slots.

See [orchestrator.md](orchestrator.md) for submission and dependency inference,
and [worker-manager.md](worker-manager.md) for endpoint execution.

## 1. Queue topology

The final queue layout separates directed NEXT_LEVEL work from freely
scheduled SUB work:

```text
NextLevelReadyQueues
├── group FIFO
└── stable worker id -> single-task FIFO

ReadyQueue
└── shared SUB FIFO

Scheduler
└── completion FIFO
```

`Orchestrator::enqueue_ready` is the only READY router:

```text
NEXT_LEVEL single -> next_level.single[target_worker_id]
NEXT_LEVEL group  -> next_level.group
SUB               -> ready_sub
```

Both immediately-ready submissions and dependency-released consumers use
this function. A directed task therefore cannot re-enter a shared
NEXT_LEVEL queue after waiting in PENDING.

Each `ReadyQueue` is a mutex-protected non-blocking FIFO. Root submission,
worker completion, and stop requests notify the Scheduler condition variable;
its wait predicate checks the completion FIFO, every ready queue, and the stop
flag. Ready queues have no blocking pop or shutdown state.

## 2. Scheduler loop

The Scheduler drains completions before dispatching new work:

```cpp
while (true) {
    wait_until_completion_ready_or_stop();

    while (completion_queue has an item) {
        on_task_complete(item);
    }

    dispatch_next_level_group();
    dispatch_next_level_singles();
    dispatch_sub_ready();

    if (stop_requested && all workers are idle) {
        drain final completions;
        dispatch one final pass;
        break;
    }
}
```

`loop_mutex` covers completion and dispatch slot access.
`Orchestrator::release_run` uses the same mutex during optional globally
quiescent compaction, preventing slot removal while the Scheduler holds a
reference.

## 3. Directed NEXT_LEVEL dispatch

### Single tasks

Every single task contains one exact stable `target_worker_id`. For each
registered NEXT_LEVEL worker, the Scheduler checks only that worker's FIFO:

```text
if target worker is idle and its FIFO is non-empty:
    pop FIFO head
    READY -> RUNNING
    dispatch to that exact worker
```

There is no idle-worker search, rebinding, work stealing, or scan into another
worker's queue. FIFO is independent per worker, so a busy worker A does not
block READY work for worker B.

### Group tasks

A group is one DAG node with one exact stable worker ID per member. The
Scheduler examines only the group FIFO head:

```text
resolve every target worker
if every target is idle:
    pop the group
    initialize all member states
    READY -> RUNNING
    dispatch member i to target_worker_ids[i]
else:
    leave the group at the FIFO head
```

The check is all-or-nothing: a blocked group reserves no partial worker set.
The Scheduler does not scan later groups. It continues to single-task queues,
so the runtime adds no fairness, aging, priority, or reservation policy beyond
trying a launchable group before singles in each iteration. Users are
responsible for choosing worker sets that make acceptable progress.

## 4. SUB dispatch

SUB has no public worker-ID selection. All READY SUB tasks share one FIFO.
For a single task the Scheduler chooses any idle SUB worker. For a group it
chooses the required number of distinct idle SUB workers; if there are not
enough, it returns the task to the queue and stops that drain pass.

This is intentionally separate from NEXT_LEVEL placement. SUB free scheduling
does not provide a compatibility path for omitted NEXT_LEVEL targets.

## 5. Dependency release and PENDING updates

Submission records each live producer in the consumer's `fanin_count`. A
consumer with live producers enters PENDING and is not placed on a ready
queue. When a producer completes successfully, the Scheduler increments each
consumer's `fanin_released`:

```cpp
if (++consumer.fanin_released >= consumer.fanin_count) {
    if (CAS(consumer.state, PENDING, READY)) {
        enqueue_ready_cb(consumer_slot);
    }
}
```

The compare-and-swap ensures that exactly one producer completion performs
the PENDING-to-READY transition. The callback routes by the consumer's own
type and exact target, not by the producer.

If a producer fails, `poison_task` moves not-yet-running downstream consumers
to FAILED instead of READY. Poisoned tasks are never dispatched but still
release references and retire normally.

## 6. Completion and group aggregation

Each `WorkerThread` reports a `WorkerCompletion` containing the slot,
`group_index`, outcome, and error message. A single-task completion goes
directly to the Scheduler completion FIFO.

Group members update per-member terminal state under `group_mu`. Only when all
members are terminal does `worker_done` enqueue one aggregate completion for
the group slot. The first member failure determines the stored failure;
members already running finish, and not-yet-dispatched SUB members are marked
skipped. Directed NEXT_LEVEL groups are launched as a complete set.

On aggregate completion the Scheduler:

1. moves the slot to COMPLETED or FAILED;
2. releases or poisons downstream consumers;
3. releases references held on this task's producers;
4. calls `try_consume` when all fanout references are released.

The Orchestrator's consume callback erases TensorMap entries that still point
to the slot, releases its Ring storage, and decrements the active-task count.

## 7. Lifecycle and invariants

`Worker::init` registers and starts WorkerManager endpoints first, freezes the
stable NEXT_LEVEL worker-ID queue map, initializes the Orchestrator, and then
starts the Scheduler. `Scheduler::stop` requests termination, wakes the loop,
waits for workers to become idle, drains final completions, and joins its
thread.

The scheduling invariants are:

1. A NEXT_LEVEL slot always has exactly one target per member.
2. A NEXT_LEVEL single is present only in its target worker's FIFO.
3. A NEXT_LEVEL group is present only in the group FIFO and launches only on
   its complete target set.
4. SUB slots never carry target-worker metadata.
5. Only the Scheduler calls `WorkerThread::dispatch`.
6. Only one successful PENDING-to-READY transition enqueues a consumer.
7. A group produces one aggregate DAG completion regardless of member count.

## 8. Related documents

- [hierarchical_level_runtime.md](hierarchical_level_runtime.md)
- [orchestrator.md](orchestrator.md)
- [worker-manager.md](worker-manager.md)
- [task-flow.md](task-flow.md)
