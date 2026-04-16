# Worker Manager — Pool, Threading, and Dispatch Modes

> **Status**: describes the **target** design. Current code still has
> separate `DistChipProcess` / `DistSubWorker` classes (target: merged
> into `WorkerThread` in PR-D). `IWorker::run(uint64_t callable,
> TaskArgsView args, const ChipCallConfig &config)` is the live
> signature; no `WorkerPayload` dispatch carrier exists. See
> [roadmap.md](roadmap.md) for the full landed-vs-planned breakdown.

`WorkerManager` and `WorkerThread` together implement the **execution layer**
of a `Worker` engine. `WorkerManager` owns two pools of `WorkerThread`s (one
for next-level workers, one for sub workers); each `WorkerThread` owns an
`IWorker` and a `std::thread`, and dispatches to it in either THREAD or
PROCESS mode.

For the high-level role of this layer among the three engine components, see
[distributed_level_runtime.md](distributed_level_runtime.md). For what the
`IWorker` implementations actually do with task data, see
[task-flow.md](task-flow.md). For where dispatched tasks come from, see
[scheduler.md](scheduler.md).

---

## 1. `WorkerManager`

```cpp
class WorkerManager {
public:
    enum class Mode { THREAD, PROCESS };

    explicit WorkerManager(Mode mode);

    // Registration (before init)
    void add_next_level(IWorker *worker);
    void add_sub       (IWorker *worker);

    // Lifecycle
    void start(OnCompleteFn on_complete);   // starts all WorkerThreads
    void stop();

    // Scheduler API
    WorkerThread *pick_idle(WorkerType type);
    std::vector<WorkerThread *> pick_n_idle(WorkerType type, int n);
    void dispatch(WorkerThread *wt, TaskSlot slot_id);

private:
    Mode mode_;
    std::vector<std::unique_ptr<WorkerThread>> next_level_;
    std::vector<std::unique_ptr<WorkerThread>> sub_;
};
```

### Responsibilities

- **Pool ownership**: two `std::vector` pools, sized at init from `add_*`
  calls
- **Idle selection**: `pick_idle(type)` finds a WorkerThread whose queue is
  empty; blocks if none available
- **Mode propagation**: every `WorkerThread` constructed under this manager
  inherits `mode_` (picked per `Worker` at construction)

### Mode choice

| Deployment | Recommended mode |
| ---------- | ---------------- |
| Onboard real hardware | `THREAD` — driver is thread-safe per device, no fork overhead |
| Simulation (sim runtime) | `PROCESS` — sim backend has shared state that needs isolation |
| `ci.py` parallel tests | `PROCESS` — test independence; per-test dlopen state |
| L4+ when L3 children are thread-safe composites | `THREAD` |

Mode is a per-`Worker` decision. Different levels in a nested hierarchy can
use different modes independently (e.g., L4 THREAD containing L3 PROCESS).

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
    enum class Mode { THREAD, PROCESS };

    WorkerThread(Mode mode,
                 IWorker *worker,
                 DistRing *ring,
                 size_t mailbox_size = 0);

    void start(OnCompleteFn on_done);
    void stop();
    void dispatch(WorkerDispatch d);       // slot id + group sub-index
    bool is_idle() const;

private:
    Mode mode_;
    IWorker *worker_;
    DistRing *ring_;                       // reads slot state via ring->slot_state(id)
    std::thread parent_thread_;
    LockFreeQueue<WorkerDispatch> queue_;

    // PROCESS mode only
    void *mailbox_ = nullptr;              // shm
    pid_t child_pid_ = -1;
    size_t mailbox_size_ = 0;

    void loop();
    void dispatch_thread(WorkerDispatch d);
    void dispatch_process(WorkerDispatch d);
    [[noreturn]] void child_loop();
    void fork_child();
};
```

The WorkerThread's `std::thread` always exists regardless of mode — it pumps
the internal queue and either runs the worker in-process or drives the shm
handshake to a forked child.

`WorkerDispatch` carries only `{slot_id, group_index}`; the thread reads
`slot.callable` / `slot.task_args` / `slot.config` on each dispatch via
`ring->slot_state(slot_id)`. For a group slot with `group_size() == N`,
the Scheduler pushes N `WorkerDispatch` entries (one per member) onto N
idle threads; each thread's `group_index` selects which
`task_args_list[i]` view to hand to the worker. There is no
`WorkerPayload` — the per-dispatch carrier is just the slot id plus the
group sub-index.

---

## 3. THREAD mode

The simple case: same process, no shm, no serialization.

```cpp
void WorkerThread::dispatch_thread(WorkerDispatch d) {
    TaskSlotState &s = *ring_->slot_state(d.task_slot);
    uint64_t callable = (s.worker_type == WorkerType::SUB)
        ? static_cast<uint64_t>(s.callable_id)      // registry id for sub path
        : s.callable;                                // ChipCallable buffer ptr
    TaskArgsView view = s.args_view(d.group_index); // 0 for single tasks
    worker_->run(callable, view, s.config);
    on_complete_(d.task_slot);
}
```

- `slot.args_view(i)` returns a zero-copy `TaskArgsView` over
  `task_args` (single) or `task_args_list[i]` (group member). The backing
  `std::vector` lives on the parent heap until the slot reaches CONSUMED.
- `callable` unifies the two registry shapes: `uint64 = ChipCallable*`
  for next-level, `uint64 = callable_id` for sub. The receiving
  `IWorker` casts back appropriately.
- `IWorker::run` dispatches polymorphically based on the actual worker type.

Note: SUB workers in PROCESS mode bypass `IWorker` entirely — the Python
child loop (``_sub_worker_loop``) reads the args blob from the mailbox,
decodes it into a ``TaskArgs``, and calls the registered callable as
``fn(args)``. The C++ dispatch path writes the same mailbox format for
both worker types.

**When is THREAD mode safe?**

- The IWorker implementation must be thread-safe relative to other concurrent
  calls and other system state
- `ChipWorker` (dlsym'd runtime.so) is safe when the runtime `.so` and its
  device driver support concurrent use
- SUB workers run in Python child processes (PROCESS mode) where the
  callable receives ``TaskArgs`` as its sole argument

---

## 4. PROCESS mode

Each WorkerThread forks a child at init. Each dispatch encodes task data into
a shm mailbox, signals the child, and polls for completion.

### 4.1 Fork at init

`fork_child()` is called once by `WorkerThread::start()` **before any C++
worker thread spawns**:

```cpp
void WorkerThread::fork_child() {
    // Alloc mailbox in MAP_SHARED shm
    mailbox_ = mmap(nullptr, mailbox_size_,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    // Initialize mailbox state to IDLE
    write_state(mailbox_, MailboxState::IDLE);

    pid_t pid = fork();
    if (pid == 0) {
        // Child
        child_loop();   // never returns
    } else {
        child_pid_ = pid;
    }
}
```

### 4.2 Fork ordering invariant

**Fork must happen before any `std::thread` is created in the parent.** The
Python `Worker` ensures this by:

1. `Worker.register(fn)` registers Python callables (pre-fork)
2. C++ `WorkerManager::add_*` registers IWorker pointers
3. `Worker::init`:
   - First: `WorkerManager::start()` — this calls each WorkerThread's
     `start`, which **forks the child, then spawns the parent's
     std::thread** for that WT
   - Then: `Scheduler::start()` spawns the scheduler thread
4. Fork ordering: at the moment `fork()` is called, the parent has only the
   Python main thread and zero C++ worker threads. Safe.

This avoids the classical "fork in multithreaded process" hazard where a
child inherits locks held by threads that don't exist post-fork.

### 4.3 Parent-side dispatch

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

### 4.4 Child loop

```cpp
void WorkerThread::child_loop() {
    for (;;) {
        while (read_state(mailbox_) != MailboxState::TASK_READY)
            pause_short();

        if (read_state(mailbox_) == MailboxState::SHUTDOWN) exit(0);

        uint8_t *d = (uint8_t*)mailbox_ + HEADER_SIZE;
        Callable     cb     = *reinterpret_cast<Callable*>(d);
        CallConfig   config = *reinterpret_cast<CallConfig*>(d + 8);
        TaskArgsView view   = read_blob(d + 8 + sizeof(CallConfig));

        int err = 0;
        try {
            worker_->run(cb, view, config);
        } catch (...) {
            err = 1;
        }
        write_error(mailbox_, err);
        write_state(mailbox_, MailboxState::TASK_DONE);
    }
}
```

Child's `worker_` is polymorphic (ChipWorker / SubWorker / nested Worker).
The child inherits the parent's full address space at fork time, so:

- ChipCallable objects (pre-fork allocated) are COW-visible at the same VA
- `py_registry` (for SubWorker) is COW-visible
- Tensor data in `torch.share_memory_()` regions is fully shared (MAP_SHARED)

### 4.5 Mailbox layout

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

### 4.6 Shutdown

```cpp
void WorkerThread::stop() {
    if (mode_ == Mode::PROCESS) {
        write_state(mailbox_, MailboxState::SHUTDOWN);
        waitpid(child_pid_, nullptr, 0);
        munmap(mailbox_, mailbox_size_);
    }
    // Signal parent thread to exit its loop
    queue_.push_sentinel();
    parent_thread_.join();
}
```

---

## 5. Adding a new IWorker type

To add a new worker kind (e.g., a RemoteWorker over RPC):

1. Implement `IWorker::run(Callable, TaskArgsView, const CallConfig&)` on the
   new class
2. Register via `manager.add_next_level(ptr)` or `manager.add_sub(ptr)`
3. If the new worker needs to run in PROCESS mode, ensure any resources it
   needs (shm regions, sockets) are established before fork

The dispatch path (THREAD vs PROCESS) is chosen by `WorkerManager::mode_`,
not by the IWorker type — so the same IWorker implementation works in both
modes. This is why `ChipWorker`, `SubWorker`, and `Worker` all share one
interface: the dispatch layer is orthogonal to the worker semantics.

---

## 6. Why this layering

Three decisions that led here:

### 6.1 Why not fork per task?

Forking per submit eliminates the mailbox and serialization, but costs
~1-10 ms per fork (COW page-table setup for a large parent image). For
thousands of tasks per DAG, the overhead dominates. Pre-forked pool amortizes
fork across many dispatches.

### 6.2 Why slot pool on parent heap, not shm?

The scheduling state (TaskSlotState.fanin_count, fanout_consumers,
fanout_mu) is parent-only — Scheduler and Orchestrator read/write it, but
children never do. Putting the slot in shm would force cross-process atomics
and shm-safe containers for no benefit. See
[task-flow.md](task-flow.md) §11 for full rationale.

### 6.3 Why one WorkerThread per IWorker?

Alternative: N workers share one dispatch queue. Rejected because:

- `WorkerThread` queue is the natural unit of backpressure — if worker `i` is
  slow, its queue fills up and scheduler falls back to another
- Simpler mental model: one IWorker = one thread that drives it
- Zero contention on queue access (only one producer, one consumer per queue)

---

## 7. Related

- [distributed_level_runtime.md](distributed_level_runtime.md) — where this
  layer fits in the three-component engine
- [task-flow.md](task-flow.md) — what `IWorker::run` receives
- [scheduler.md](scheduler.md) — the producer of `WorkerThread::dispatch`
  calls
