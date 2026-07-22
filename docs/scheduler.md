# Scheduler — DAG Dispatch Internals

The Scheduler is the **DAG executor**. A dedicated C++ thread that consumes
submitted slots, wires fanout edges, dispatches ready tasks to worker threads,
and handles completion callbacks. It is the bridge between the Orchestrator
(producer of DAG nodes) and the WorkerManager (consumer of ready nodes).

For the high-level role of the Scheduler among the three engine components,
see [hierarchical_level_runtime.md](hierarchical_level_runtime.md). For the DAG
construction side (what feeds the Scheduler), see
[orchestrator.md](orchestrator.md). For dispatch mechanics (how
`WorkerThread::dispatch` actually runs a task), see
[worker-manager.md](worker-manager.md).

---

## 1. Role

The Scheduler's job:

- Drain the **wiring queue** (Phase 0): wire fanout edges for newly
  submitted slots; if all producers are already done, promote to the ready queue.
- Drain the **ready queues** (Phase 1): dispatch each NEXT_LEVEL single task
  only to its requested worker; group and SUB dispatch remain on their shared
  queues during this transition.
- Drain the **completion queue** (Phase 2): for each worker completion,
  transition the slot to `COMPLETED` or `FAILED`, release fanout references,
  wake or poison downstream consumers, and (if all refs released) retire the
  ring slot.

One Scheduler per `Worker` instance, one thread per Scheduler. The Scheduler
**does not inspect task data** — it moves slot ids between queues and
consults scheduling metadata (`fanin_count`, `fanout_consumers`, `state`).

---

## 2. The queues

```cpp
class Scheduler {
    // Producer: Orchestrator.submit_*. Consumer: Scheduler's own loop, Phase 0.
    LockFreeQueue<WiringEntry> wiring_queue_;       // {slot, producers}

    // Owns one single-task FIFO per stable worker id and one group FIFO.
    NextLevelReadyQueues *ready_next_level_queues_;

    // SUB scheduling stays unconstrained and shared.
    ReadyQueue *ready_sub_queue_;

    // Shared READY router used by submit, wiring, and dependency release.
    std::function<void(TaskSlot)> enqueue_ready_cb_;

    // Producer: WorkerThread (on endpoint->run() return).
    // Consumer: Scheduler's own loop, Phase 2.
    std::queue<WorkerCompletion> completion_queue_;
};
```

### Wiring queue

Introduced so that `Orchestrator::submit_*` does not need to acquire
`fanout_mu` on every producer slot at submit time (see
[orchestrator.md](orchestrator.md) §2 step 6).

Each entry:

```cpp
struct WiringEntry {
    TaskSlot consumer;
    std::vector<TaskSlot> producers;    // producers this consumer depends on
};
```

### Ready queue

Slots whose `fanin_count == fanin_released` are ready to dispatch. The queue
holds just the slot id; dispatch reads task data from the
`ring.slot_state(sid)` pool.

NEXT_LEVEL single tasks have one FIFO for every stable worker id. A task is
inserted into the FIFO named by its required `worker` argument:

```cpp
NextLevelReadyQueues ready_next_level_queues_; // worker FIFOs + group FIFO
ReadyQueue ready_sub_queue_;                   // all SUB tasks
```

The Orchestrator owns the routing rule. Both immediately-ready submissions
and Scheduler transitions to READY call the same `enqueue_ready` entry point,
so dependency release cannot accidentally return a directed single task to a
shared queue. A busy target blocks only its own FIFO; the Scheduler continues
checking other worker FIFOs. FIFO order is preserved independently per worker.

NEXT_LEVEL groups use the dedicated group FIFO. The Scheduler examines only
its head. A group leaves READY only when every submitted target worker is idle;
otherwise the group stays at the head, no worker is reserved, and independent
single-task queues can still dispatch. SUB dispatch remains unchanged.

### Completion queue

Endpoint completions whose worker returned or failed. Each
`WorkerCompletion` carries `{slot, group_index, outcome, error_message}`;
`outcome` is success, task failure, endpoint failure, or skipped. The
Scheduler runs completion handling (fanout release, downstream wake/poison,
try_consume) in its own thread so that WorkerThreads can immediately return to
their next task.

---

## 3. Scheduler loop (pseudocode)

```cpp
void Scheduler::run() {
    while (running_) {
        // Phase 0: wiring
        WiringEntry w;
        while (wiring_queue_.try_pop(w)) {
            wire_fanout(w);   // see §4
        }

        // Phase 1: dispatch (drains BOTH per-type queues; see §5)
        dispatch_ready();

        // Phase 2: completion
        WorkerCompletion c;
        while (completion_queue_.try_pop(c)) {
            on_task_complete(c);   // see §6
        }

        // If all three queues empty, block on a condition variable until
        // any producer signals work.
        wait_for_work();
    }
}
```

Phase order matters:

- Wiring before dispatch: a task may become ready during wiring (all its
  producers already completed); wiring promotes it to ready_queue in the
  same Scheduler iteration.
- Dispatch before completion: dispatch the backlog first to keep workers
  busy; completion handling is not time-critical (fanout release just
  queues more work for the next iteration).

---

## 4. Phase 0 — wiring

```cpp
void Scheduler::wire_fanout(const WiringEntry &w) {
    TaskSlot csid = w.consumer;
    TaskSlotState &c = slots_[csid];
    int32_t actual_live = 0;

    for (TaskSlot psid : w.producers) {
        TaskSlotState &p = slots_[psid];
        std::lock_guard lk(p.fanout_mu);
        // COMPLETED producers are already done; FAILED producers poison this
        // consumer instead of making it ready.
        if (p.state.load() == TaskState::COMPLETED ||
            p.state.load() == TaskState::CONSUMED) continue;
        if (p.state.load() == TaskState::FAILED) {
            poison_task(csid, p.failure_message);
            continue;
        }
        p.fanout_consumers.push_back(csid);
        p.fanout_total++;
        actual_live++;
    }

    // Update consumer's fanin to the actual live count (producers already
    // finished don't count).
    c.fanin_count = actual_live;
    if (actual_live == 0) {
        enqueue_ready_cb_(csid);
    }
}
```

**Race with completion**: a producer may finish between submit and wiring.
The `lock_guard(p.fanout_mu)` + `p.state.load()` check ensures we either:

- wire an edge and the producer's future completion will fire `fanin_released++`
  for this consumer, or
- see "already completed" and skip, correctly counting this producer as not
  contributing to fanin.

---

## 5. Phase 1 — dispatch

`dispatch_ready` tries launchable NEXT_LEVEL groups first, then checks each
NEXT_LEVEL worker's single-task FIFO, and finally drains the SUB queue. The
only cross-queue policy is launchable-group-first.

Group dispatch is all-or-nothing:

```cpp
TaskSlot group;
if (ready_next_level_queues_->try_front_group(group)) {
    resolve every submitted worker id;
    if (every target worker is idle) {
        ready_next_level_queues_->try_pop_group(group);
        mark every group member RUNNING;
        dispatch every member to its submitted target;
    }
}
```

If any target is busy, the head is not removed or moved to the tail. The
Scheduler does not scan later groups and does not reserve the currently idle
members. It continues to the single-task queues:

```cpp
void Scheduler::dispatch_next_level_singles() {
    for (int32_t worker_id : ready_next_level_queues_->worker_ids()) {
        WorkerThread *worker =
            manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
        if (!worker || !worker->idle()) continue;

        TaskSlot slot;
        while (worker->idle() &&
               ready_next_level_queues_->try_pop_single(worker_id, slot)) {
            TaskSlotState &task = slots_[slot];
            if (task.state.load() != TaskState::READY) continue;
            task.state.store(TaskState::RUNNING);
            worker->dispatch({slot, 0});
        }
    }
}
```

Dispatch hands off a `WorkerDispatch {slot, group_index}` to a
`WorkerThread`. The WorkerThread reads
`ring.slot_state(slot).{callable, task_args, config}` on its own thread
and encodes it into the per-WT mailbox — see
[worker-manager.md](worker-manager.md) §3 for the dispatch protocol.

**Directed-single back-pressure**: if the requested worker is busy, its FIFO
is left untouched. Other NEXT_LEVEL worker FIFOs and the SUB queue still make
progress. The ring's back-pressure at the Orch side caps the total number of
in-flight tasks.

Worker eligibility is opaque scheduling metadata. The Scheduler compares
worker ids and capability bits exposed through `WorkerEndpoint::caps()`, but
does not inspect HCOMM, RDMA, socket, or remote buffer internals.
For a NEXT_LEVEL single task, `s.get_affinity(0)` is the stable requested
worker id and can differ from the `next_level_threads_` vector index. SUB has
no public affinity.

---

## 6. Phase 2 — completion

Called by `WorkerThread::on_complete_(completion)` which pushes to
`completion_queue_`. The Scheduler then:

```cpp
void Scheduler::on_task_complete(const WorkerCompletion &completion) {
    TaskSlot sid = completion.task_slot;
    TaskSlotState &s = slots_[sid];

    // Group tasks aggregate per-member outcomes before the slot is terminal.
    if (s.group_size > 0) {
        if (!record_group_member_completion(completion)) return;
    }

    bool failed = completion.outcome != EndpointOutcome::SUCCESS;
    s.state.store(failed ? TaskState::FAILED : TaskState::COMPLETED);

    // Release fanout refs on downstream consumers
    std::vector<TaskSlot> consumers;
    {
        std::lock_guard lk(s.fanout_mu);
        consumers = s.fanout_consumers;    // snapshot (mutex protects vector)
    }
    for (TaskSlot csid : consumers) {
        if (failed) {
            poison_task(csid, completion.error_message);
            continue;
        }
        TaskSlotState &c = slots_[csid];
        if (++c.fanin_released == c.fanin_count) {
            enqueue_ready_cb_(csid);
        }
    }

    // Also: this task itself may now be CONSUMED
    try_consume(sid);
}
```

### `try_consume`

```cpp
void Scheduler::try_consume(TaskSlot sid) {
    TaskSlotState &s = slots_[sid];
    if (s.state.load() != TaskState::COMPLETED &&
        s.state.load() != TaskState::FAILED) return;
    if (s.fanout_released.load() != s.fanout_total) return;

    s.state.store(TaskState::CONSUMED);

    // Erase tensormap entries this task produced
    for (int i = 0; i < s.task_args.tensor_count(); i++) {
        // only erase entries still pointing at this slot
        uint64_t ptr = s.task_args.tensor(i).data;
        if (orchestrator_->tensormap_lookup(ptr) == sid)
            orchestrator_->tensormap_erase(ptr);
    }

    // Return slot to ring pool
    ring_->release(sid);
    s.state.store(TaskState::FREE);
}
```

Scope release (when `scope_end` runs) calls back into the Scheduler to bump
`fanout_released` by 1 on each scope-registered slot, triggering
`try_consume`. This is how leaf tasks get reclaimed.

---

## 7. Start / Stop

```cpp
void Scheduler::start(Config cfg) {
    manager_ = cfg.manager;
    orchestrator_ = cfg.orchestrator;
    running_.store(true);
    thread_ = std::thread([this] { run(); });
}

void Scheduler::stop() {
    running_.store(false);
    wake();
    thread_.join();
}
```

`Worker::init` calls `start` after all children are registered and
`WorkerManager::start` has spawned the WorkerThread pool.

---

## 8. Completion channel from WorkerThread

```cpp
// In WorkerThread, after endpoint->run() returns:
void WorkerThread::loop() {
    for (;;) {
        TaskSlot sid = queue_.pop();
        WorkerCompletion c = endpoint_->run(ring_, {sid, group_index});
        scheduler_->completion_queue_.push(c);   // notify Scheduler
    }
}
```

The completion path is one-way and asynchronous: the WorkerThread returns to
its own queue immediately, and the Scheduler handles completion in its own
loop. This keeps worker dispatch latency bounded by dispatch cost alone, not
by completion-handling cost.

---

## 9. Invariants

1. **Scheduler is single-threaded**: all three phase handlers run in the
   Scheduler's own thread. Atomics/mutexes on slot state are only needed for
   Orch/WorkerThread ↔ Scheduler coordination.
2. **Slot transitions are monotonic**: success follows
   `FREE → PENDING → READY → RUNNING → COMPLETED → CONSUMED`; failure follows
   `FREE → PENDING/READY/RUNNING → FAILED → CONSUMED`.
3. **Dispatch consumes one ready entry**: every `ready_queue.push` is
   matched by exactly one `pick_idle + dispatch`. Group tasks push once,
   dispatch N times via `pick_n_idle`.
4. **Completion is per-worker for groups**: `worker_done` is called
   `group_size` times; only the terminal aggregate pushes one slot completion.
   If any member fails, not-yet-dispatched members become skipped and already
   running members are allowed to finish.
5. **Failed producers poison consumers**: consumers of a failed producer move
   to `FAILED`, are never dispatched, and still run normal cleanup.
6. **`try_consume` is idempotent on CONSUMED**: a repeated call after
   CONSUMED is a no-op.

---

## 10. Related

- [hierarchical_level_runtime.md](hierarchical_level_runtime.md) — high-level
  three-component picture
- [orchestrator.md](orchestrator.md) — the producer feeding the wiring queue
- [worker-manager.md](worker-manager.md) — where dispatched slots go
- [task-flow.md](task-flow.md) — the data (Callable / TaskArgs / CallConfig)
  that the Scheduler moves around, opaquely, by slot id
