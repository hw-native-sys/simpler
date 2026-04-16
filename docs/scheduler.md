# Scheduler — DAG Dispatch Internals

> **Status**: target design. `IWorker::run(uint64_t callable, TaskArgsView,
> ChipCallConfig)` is the live dispatch signature; per-worker-type ready
> queue split (Strict-4) is landed — each `WorkerType` has its own
> `DistReadyQueue` and `dispatch_ready` walks them independently, so a
> saturated pool of one type cannot head-of-line-block dispatch for the
> other. See [roadmap.md](roadmap.md) for the full landed-vs-planned
> breakdown.

The Scheduler is the **DAG executor**. A dedicated C++ thread that consumes
submitted slots, wires fanout edges, dispatches ready tasks to worker threads,
and handles completion callbacks. It is the bridge between the Orchestrator
(producer of DAG nodes) and the WorkerManager (consumer of ready nodes).

For the high-level role of the Scheduler among the three engine components,
see [distributed_level_runtime.md](distributed_level_runtime.md). For the DAG
construction side (what feeds the Scheduler), see
[orchestrator.md](orchestrator.md). For dispatch mechanics (how
`WorkerThread::dispatch` actually runs a task), see
[worker-manager.md](worker-manager.md).

---

## 1. Role

The Scheduler's job:

- Drain the **wiring queue** (Phase 0): wire fanout edges for newly
  submitted slots; if all producers are already done, promote to the ready queue.
- Drain the **ready queue** (Phase 1): for each ready slot, pick an idle
  `WorkerThread` from the appropriate pool and hand off.
- Drain the **completion queue** (Phase 2): for each completed slot, release
  its fanout references, wake downstream consumers, and (if all refs
  released) retire the ring slot.

One Scheduler per `Worker` instance, one thread per Scheduler. The Scheduler
**does not inspect task data** — it moves slot ids between queues and
consults scheduling metadata (`fanin_count`, `fanout_consumers`, `state`).

---

## 2. The queues

```cpp
class Scheduler {
    // Producer: Orchestrator.submit_*. Consumer: Scheduler's own loop, Phase 0.
    LockFreeQueue<WiringEntry> wiring_queue_;       // {slot, producers}

    // Strict-4 — per-worker-type ready queues.
    // Producers: Orchestrator.submit_* (routes by slot.worker_type) +
    //            Scheduler Phase 0 / Phase 2 (fanout-released; routes by
    //            consumer worker_type).
    // Consumer: Scheduler's own loop, Phase 1 (one drain loop per queue,
    //           each with its own head-of-line break).
    DistReadyQueue *ready_next_level_queue_;
    DistReadyQueue *ready_sub_queue_;

    // Producer: WorkerThread (on worker->run() return).
    // Consumer: Scheduler's own loop, Phase 2.
    LockFreeQueue<TaskSlot> completion_queue_;
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

**Strict-4 — per-worker-type split.** In practice the ready queue is two
`DistReadyQueue` instances, one per `WorkerType`:

```cpp
DistReadyQueue ready_next_level_queue_;   // WorkerType::NEXT_LEVEL tasks
DistReadyQueue ready_sub_queue_;          // WorkerType::SUB tasks
```

Matching L2's per-shape ready buffer (`PTO2_LocalReadyBuffer` fan-out to
AIC / AIV / MIX queues), with the L3+ exception that we use `std::queue`
(Allowed Exception 3: dynamic data structures on host) and only two
worker types (Allowed Exception 2: `NEXT_LEVEL` + `SUB` at L3+, not
AIC / AIV / MIX). `Orchestrator::submit_*` routes each slot to the queue
matching `slot.worker_type`; `Scheduler::on_task_complete` routes a
newly-ready consumer the same way, based on the *consumer's* worker
type. `Scheduler::dispatch_ready` drains each queue with its own
head-of-line break so a saturated pool of one type cannot stall dispatch
for the other.

### Completion queue

Slots whose worker returned. The Scheduler runs completion handling
(fanout release, downstream wake, try_consume) in its own thread so that
WorkerThreads can immediately return to their next task.

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
        while (completion_queue_.try_pop(sid)) {
            on_task_complete(sid);   // see §6
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
        // If producer has already reached COMPLETED/CONSUMED, its fanout is
        // already finalized — consumer sees it as "done", no edge to add.
        if (p.state.load() >= TaskState::COMPLETED) continue;
        p.fanout_consumers.push_back(csid);
        p.fanout_total++;
        actual_live++;
    }

    // Update consumer's fanin to the actual live count (producers already
    // finished don't count).
    c.fanin_count = actual_live;
    if (actual_live == 0) {
        // Strict-4: wiring promotes directly to the per-type queue.
        auto *q = (c.worker_type == WorkerType::NEXT_LEVEL) ? ready_next_level_queue_
                                                            : ready_sub_queue_;
        q->push(csid);
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

`dispatch_ready` drains each per-type ready queue with its own
head-of-line break so one saturated pool cannot stall the other:

```cpp
void Scheduler::dispatch_ready() {
    auto drain_one = [&](DistReadyQueue *q) {
        DistTaskSlot slot;
        while (q->try_pop(slot)) {
            TaskSlotState &s = slots_[slot];
            int N = s.group_size();  // 1 for single-task slots

            auto workers = manager_->pick_n_idle(s.worker_type, N);
            if (static_cast<int>(workers.size()) < N) {
                q->push(slot);   // put back; try again after a completion
                break;
            }
            s.state.store(TaskState::RUNNING);
            for (int i = 0; i < N; i++) {
                workers[i]->dispatch({slot, i});
            }
        }
    };
    drain_one(ready_next_level_queue_);
    drain_one(ready_sub_queue_);
}
```

Dispatch hands off a `WorkerDispatch {slot, group_index}` to a
`WorkerThread`. The WorkerThread reads
`ring.slot_state(slot).{callable, task_args, config}` on its own thread
and executes — see [worker-manager.md](worker-manager.md) §3 for THREAD
mode and §4 for PROCESS mode.

**Pick-idle back-pressure**: when `pick_n_idle` returns fewer workers
than the task needs, the slot is pushed back onto *its* queue and that
queue's drain halts; the other-type queue's drain continues. The ring's
back-pressure at the Orch side already caps the total number of
in-flight tasks across both types.

---

## 6. Phase 2 — completion

Called by `WorkerThread::on_complete_(sid)` which pushes to
`completion_queue_`. The Scheduler then:

```cpp
void Scheduler::on_task_complete(TaskSlot sid) {
    TaskSlotState &s = slots_[sid];

    // Group tasks require all sub-workers to finish
    if (s.group_size > 0) {
        if (s.sub_complete_count.fetch_add(1) + 1 < s.group_size) return;
    }

    s.state.store(TaskState::COMPLETED);

    // Release fanout refs on downstream consumers
    std::vector<TaskSlot> consumers;
    {
        std::lock_guard lk(s.fanout_mu);
        consumers = s.fanout_consumers;    // snapshot (mutex protects vector)
    }
    for (TaskSlot csid : consumers) {
        TaskSlotState &c = slots_[csid];
        if (++c.fanin_released == c.fanin_count) {
            // Strict-4: push to the queue matching the *consumer's*
            // worker type. A consumer of a NEXT_LEVEL producer can itself
            // be SUB, so we pick based on `c.worker_type`, not `s`.
            auto *q = (c.worker_type == WorkerType::NEXT_LEVEL) ? ready_next_level_queue_
                                                                : ready_sub_queue_;
            q->push(csid);
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
    if (s.state.load() != TaskState::COMPLETED) return;
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
// In WorkerThread, after worker->run() returns:
void WorkerThread::loop() {
    for (;;) {
        TaskSlot sid = queue_.pop();
        // ... run worker (see worker-manager.md for THREAD/PROCESS) ...
        scheduler_->completion_queue_.push(sid);   // notify Scheduler
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
2. **Slot transitions are monotonic**: `FREE → PENDING → READY → RUNNING →
   COMPLETED → CONSUMED` never reverses within one allocation.
3. **Dispatch consumes one ready entry**: every `ready_queue.push` is
   matched by exactly one `pick_idle + dispatch`. Group tasks push once,
   dispatch N times via `pick_n_idle`.
4. **Completion is per-worker for groups**: `on_task_complete` is called
   `group_size` times; only the last one triggers the actual transition.
5. **`try_consume` is idempotent on CONSUMED**: a repeated call after
   CONSUMED is a no-op.

---

## 10. Related

- [distributed_level_runtime.md](distributed_level_runtime.md) — high-level
  three-component picture
- [orchestrator.md](orchestrator.md) — the producer feeding the wiring queue
- [worker-manager.md](worker-manager.md) — where dispatched slots go
- [task-flow.md](task-flow.md) — the data (Callable / TaskArgs / CallConfig)
  that the Scheduler moves around, opaquely, by slot id
