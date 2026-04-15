# Scheduler — DAG Dispatch Internals

> **Status**: target design. `IWorker::run(uint64_t callable, TaskArgsView,
> ChipCallConfig)` is the live dispatch signature; per-worker-type ready
> queue split (Strict-4) is not yet implemented (lands in PR-D). See
> [roadmap.md](roadmap.md) for the full landed-vs-planned breakdown.

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

## 2. The three queues

```cpp
class Scheduler {
    // Producer: Orchestrator.submit_*. Consumer: Scheduler's own loop, Phase 0.
    LockFreeQueue<WiringEntry> wiring_queue_;       // {slot, producers}

    // Producer: Scheduler Phase 0 (newly-ready) + Phase 2 (fanout-released).
    // Consumer: Scheduler's own loop, Phase 1.
    LockFreeQueue<TaskSlot> ready_queue_;

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
holds just the slot id; dispatch reads task data from `slots_[sid]`.

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

        // Phase 1: dispatch
        TaskSlot sid;
        while (ready_queue_.try_pop(sid)) {
            dispatch_ready(sid);   // see §5
        }

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
    if (actual_live == 0) ready_queue_.push(csid);
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

```cpp
void Scheduler::dispatch_ready(TaskSlot sid) {
    TaskSlotState &s = slots_[sid];
    s.state.store(TaskState::READY);

    if (s.group_size == 0) {
        // Single-worker task
        WorkerThread *wt = manager_->pick_idle(s.worker_type);
        wt->dispatch(sid);
    } else {
        // Group task — reserve N idle workers
        auto wts = manager_->pick_n_idle(s.worker_type, s.group_size);
        s.state.store(TaskState::RUNNING);
        for (size_t i = 0; i < wts.size(); i++) {
            wts[i]->dispatch(sid, /*group_index=*/static_cast<int32_t>(i));
        }
    }
}
```

Dispatch hands off the slot id to a `WorkerThread`. The WorkerThread reads
`slots_[sid].{callable, task_args, config}` on its own thread and executes —
see [worker-manager.md](worker-manager.md) §3 for THREAD mode and §4 for
PROCESS mode.

**Pick-idle back-pressure**: if no idle worker exists in the pool,
`pick_idle` blocks. The Scheduler thread is then stalled, which is fine —
ready tasks pile up in the queue until a worker frees up. The ring's
back-pressure at the Orch side already caps the number of in-flight tasks.

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
            ready_queue_.push(csid);       // consumer now ready
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
