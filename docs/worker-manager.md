# Worker Manager — Pool, Threading, and Dispatch

`WorkerManager` and `WorkerThread` together implement the **execution layer**
of a `Worker` engine. `WorkerManager` owns two pools of `WorkerThread`s (one
for next-level workers, one for sub workers); each `WorkerThread` drives a
shared-memory mailbox that a forked Python child consumes — the child runs
the real `IWorker` in its own address space.

For the high-level role of this layer among the three engine components, see
[hierarchical_level_runtime.md](hierarchical_level_runtime.md). For what the
`IWorker` implementations actually do with task data, see
[task-flow.md](task-flow.md). For where dispatched tasks come from, see
[scheduler.md](scheduler.md).

---

## 1. `WorkerManager`

```cpp
class WorkerManager {
public:
    // Registration (before init). `mailbox` is a MAILBOX_SIZE-byte
    // MAP_SHARED region; the real IWorker lives in the forked child.
    void add_next_level(void *mailbox);
    void add_sub       (void *mailbox);

    // Lifecycle
    void start(Ring *ring, OnCompleteFn on_complete);   // starts all WorkerThreads
    void stop();

    // Scheduler API
    WorkerThread *pick_idle(WorkerType type) const;
    std::vector<WorkerThread *> pick_n_idle(WorkerType type, int n) const;

private:
    std::vector<void *> next_level_entries_;
    std::vector<void *> sub_entries_;
    std::vector<std::unique_ptr<WorkerThread>> next_level_threads_;
    std::vector<std::unique_ptr<WorkerThread>> sub_threads_;
};
```

### Responsibilities

- **Pool ownership**: two `std::vector` pools, sized at init from `add_*`
  calls
- **Idle selection**: `pick_idle(type)` finds a WorkerThread whose queue is
  empty; returns nullptr if none available

---

## 2. `WorkerThread`

One WorkerThread per IWorker instance.

```cpp
struct WorkerDispatch {
    TaskSlot task_slot;
    int32_t  group_index = 0;    // 0 for non-group; 0..N-1 for group members
};

class WorkerThread {
public:
    void start(Ring *ring, WorkerManager *manager,
               const std::function<void(TaskSlot)> &on_complete,
               void *mailbox);
    void stop();
    void dispatch(WorkerDispatch d);       // slot id + group sub-index
    bool idle() const;

private:
    Ring *ring_;                       // reads slot state via ring->slot_state(id)
    void *mailbox_ = nullptr;          // MAP_SHARED region the child polls
    std::thread thread_;
    std::queue<WorkerDispatch> queue_;
    std::mutex mu_;
    std::condition_variable cv_;

    void loop();
    void dispatch_process(TaskSlotState &s, int32_t group_index);
};
```

The WorkerThread's `std::thread` pumps the internal queue and drives the
shm handshake — one mailbox round trip per dispatch. The forked child loop
that consumes the mailbox lives in Python (`_chip_process_loop` /
`_sub_worker_loop` in `python/simpler/worker.py`); the parent does not fork
children.

`WorkerDispatch` carries only `{slot_id, group_index}`; the thread reads
`slot.callable` / `slot.task_args` / `slot.config` on each dispatch via
`ring->slot_state(slot_id)`. For a group slot with `group_size() == N`,
the Scheduler pushes N `WorkerDispatch` entries (one per member) onto N
idle threads; each thread's `group_index` selects which
`task_args_list[i]` view to hand to the worker. There is no
`WorkerPayload` — the per-dispatch carrier is just the slot id plus the
group sub-index.

---

## 3. Dispatch via shm mailbox

Each WorkerThread drives a `MAILBOX_SIZE`-byte `MAP_SHARED` region. The
Python facade forks one child per mailbox **before** `WorkerManager::start()`
(so the parent has only the Python main thread when fork runs, avoiding the
classical "fork in a multi-threaded process" hazard) and the child polls
the mailbox for the lifetime of the worker.

### 3.1 Parent-side dispatch

```cpp
void WorkerThread::dispatch_process(WorkerDispatch d) {
    TaskSlotState &s = *ring_->slot_state(d.task_slot);
    uint8_t *m = (uint8_t*)mailbox_ + HEADER_SIZE;

    // Write task data: callable (8 B) + config (~16 B) + length-prefixed
    // TaskArgs blob. The blob is derived from the slot's stored TaskArgs
    // via write_blob() — tags are stripped, only [T][S][tensors][scalars]
    // crosses the fork boundary.
    uint64_t callable = (s.worker_type == WorkerType::SUB)
        ? static_cast<uint64_t>(s.callable_id)
        : s.callable;
    *reinterpret_cast<uint64_t*>(m)            = callable;
    *reinterpret_cast<CallConfig*>(m + 8)      = s.config;
    const TaskArgs &args = s.is_group() ? s.task_args_list[d.group_index] : s.task_args;
    write_blob(m + 8 + sizeof(CallConfig), args);

    // Signal child
    write_state(mailbox_, MailboxState::TASK_READY);

    // Poll for completion
    while (read_state(mailbox_) != MailboxState::TASK_DONE)
        std::this_thread::sleep_for(std::chrono::microseconds(50));

    int err = read_error(mailbox_);
    write_state(mailbox_, MailboxState::IDLE);
    on_complete_(d.task_slot, err);
}
```

Parent-side cost per dispatch:

- One memcpy of `Callable` (8 B) + `CallConfig` (~16 B) + blob (≤~1.7 KB for
  L3)
- One signal (`write_state`)
- Poll loop with `sleep_for(50us)` (not busy-wait)

Total ~nanoseconds overhead; the wait is dominated by actual kernel execution.

### 3.2 Child loop

The child loop lives in Python — see `_chip_process_loop` and
`_sub_worker_loop` in `python/simpler/worker.py`. Each child polls
`MAILBOX_OFF_STATE`, decodes the args blob on `TASK_READY`, runs the
real `IWorker` (`ChipWorker` for chip children) or registered Python
callable (sub workers), writes back any error, and publishes `TASK_DONE`.
The child inherits the parent's full address space at fork time, so:

- ChipCallable objects (pre-fork allocated) are COW-visible at the same VA
- The Python callable registry is COW-visible
- Tensor data in `torch.share_memory_()` regions is fully shared (MAP_SHARED)

### 3.3 Mailbox layout

```text
offset 0:                              int32  state    (IDLE / TASK_READY / TASK_DONE / SHUTDOWN)
offset 4:                              int32  error
offset 8:                              uint64 callable
offset 16:                             CallConfig config
offset 16 + sizeof(CallConfig):        bytes  blob:  [int32 T][int32 S][ContinuousTensor × T][uint64_t × S]
```

Sized at WorkerThread construction:

```cpp
mailbox_size_ = HEADER_SIZE                    // 8 B (state + error)
              + sizeof(Callable)                // 8 B
              + sizeof(CallConfig)              // ~16 B
              + MAX_BLOB_SIZE;                  // per-level, e.g. 1672 B for L3
```

Per-worker total: ~2 KB. Typical pool: 4-8 workers → ~8-16 KB shm total.

### 3.4 Shutdown

`WorkerManager::shutdown_children()` writes `SHUTDOWN` to every registered
mailbox; each child loop sees it on its next poll and exits. The Python
facade owns the child PIDs and calls `waitpid()` after writing `SHUTDOWN`
to its own mailbox copy. The parent's `WorkerThread::stop()` only joins
the C++ dispatcher thread — it does not own the child process.

---

## 4. Adding a new IWorker type

To add a new worker kind (e.g., a RemoteWorker over RPC):

1. Implement `IWorker::run(callable, TaskArgsView, const CallConfig&)` on the
   new class
2. Decode the mailbox in a child-process loop (mirroring
   `_chip_process_loop`) so the parent talks to it through the same
   handshake as `ChipWorker`
3. Register the mailbox via `manager.add_next_level(mailbox)` or
   `manager.add_sub(mailbox)`

### 4.1 Nested fork ordering (L4+ Worker children)

When an L4 Worker has L3 Worker children (PROCESS mode), the fork sequence
nests:

```text
L4 parent process
  ├─ _init_hierarchical(): Worker(4) + HeapRing mmap (before fork)
  └─ _start_hierarchical() (on first run):
       ├─ fork L3 child  ────────►  L3 child process:
       │                              inner_worker.init()  ← Worker(3) + L3 HeapRing
       │                              _child_worker_loop()
       │                                └─ on first dispatch: inner_worker.run()
       │                                     └─ _start_hierarchical() forks L3's sub/chip children
       └─ register mailbox with L4's Worker
```

Each inner Worker inits **inside its forked child process** so its own
children are forked from the correct parent. The L4 parent never sees L3's
sub/chip grandchildren — they're L3's responsibility.

**Key invariant**: `Worker(N)` and its HeapRing are created before any
fork at level N. Children inherit the `MAP_SHARED` mmap at the same virtual
address. C++ scheduler threads start only after all forks at that level.

---

## 5. Why this layering

Three decisions that led here:

### 5.1 Why not fork per task?

Forking per submit eliminates the mailbox and serialization, but costs
~1-10 ms per fork (COW page-table setup for a large parent image). For
thousands of tasks per DAG, the overhead dominates. Pre-forked pool amortizes
fork across many dispatches.

### 5.2 Why slot pool on parent heap, not shm?

The scheduling state (TaskSlotState.fanin_count, fanout_consumers,
fanout_mu) is parent-only — Scheduler and Orchestrator read/write it, but
children never do. Putting the slot in shm would force cross-process atomics
and shm-safe containers for no benefit. See
[task-flow.md](task-flow.md) §11 for full rationale.

### 5.3 Why one WorkerThread per IWorker?

Alternative: N workers share one dispatch queue. Rejected because:

- `WorkerThread` queue is the natural unit of backpressure — if worker `i` is
  slow, its queue fills up and scheduler falls back to another
- Simpler mental model: one IWorker = one thread that drives it
- Zero contention on queue access (only one producer, one consumer per queue)

---

## 6. Related

- [hierarchical_level_runtime.md](hierarchical_level_runtime.md) — where this
  layer fits in the three-component engine
- [task-flow.md](task-flow.md) — what `IWorker::run` receives
- [scheduler.md](scheduler.md) — the producer of `WorkerThread::dispatch`
  calls
