# Worker Manager — Pool, Threading, and Dispatch

Callable identity update: WorkerThread task frames now carry the submitted
callable's 32-byte digest; the child resolves that digest to its private local
slot. Older callable-id snippets below are historical shorthand for
target-local internals. See
[callable-identity-registration.md](callable-identity-registration.md).

`WorkerManager`, `WorkerThread`, and `WorkerEndpoint` together implement the
**execution layer** of a `Worker` engine. In today's local implementation,
`WorkerManager` owns two pools of `WorkerThread`s (one for next-level workers,
one for sub workers); each `WorkerThread` owns a `LocalMailboxEndpoint` that
drives a shared-memory mailbox consumed by a forked Python child. The child
runs the real worker (a `ChipWorker` for NEXT_LEVEL, a Python callable for
SUB) in its own address space.

The remote L3 design keeps this local fork/shm path behind
`LocalMailboxEndpoint` and reserves the same `WorkerEndpoint` boundary for a
framed `RemoteL3Endpoint` for cross-host NEXT_LEVEL children. A remote endpoint
is not another child loop that polls the `MAILBOX_SIZE`-byte mailbox; it uses the
contracts in
[remote-l3-worker-design.md](remote-l3-worker-design.md).
The current code includes that `RemoteL3Endpoint` boundary, a socket-backed
simulation transport, and the daemon/session runner used by
`Worker.add_remote_worker()` for sim remote L3 endpoints. HCOMM hardware
profiles are still pending.

For the high-level role of this layer among the three engine components, see
[hierarchical_level_runtime.md](hierarchical_level_runtime.md). For what
runs on the other side of the local mailbox, see [task-flow.md](task-flow.md).
For where dispatched tasks come from, see [scheduler.md](scheduler.md).

---

## 1. `WorkerManager`

```cpp
class WorkerManager {
public:
    // Registration (before init). `mailbox` is a MAILBOX_SIZE-byte
    // MAP_SHARED region; the real worker (a `ChipWorker` for NEXT_LEVEL,
    // a Python callable for SUB) lives in the forked child.
    void add_next_level(void *mailbox);
    void add_next_level_at(int32_t worker_id, void *mailbox);
    void add_next_level_endpoint(std::unique_ptr<WorkerEndpoint> endpoint);
    void add_sub       (void *mailbox);

    // Lifecycle
    void start(Ring *ring, OnCompleteFn on_complete);   // starts all WorkerThreads
    void stop();

    // Scheduler API
    WorkerThread *get_worker_by_index(WorkerType type, int worker_index) const;
    WorkerThread *get_worker_by_id(WorkerType type, int32_t worker_id) const;
    WorkerThread *pick_idle(WorkerType type,
                            const std::vector<WorkerThread *> &exclude,
                            const std::vector<int32_t> &eligible_worker_ids) const;

private:
    struct LocalNextLevelEntry {
        int32_t worker_id;
        void *mailbox;
    };
    std::vector<LocalNextLevelEntry> next_level_entries_;
    std::vector<void *> sub_entries_;
    std::vector<std::unique_ptr<WorkerThread>> next_level_threads_;
    std::vector<std::unique_ptr<WorkerThread>> sub_threads_;
};
```

`add_next_level_at(...)` is used by the Python `Worker` facade when local L4
children share the NEXT_LEVEL worker id space with remote L3 workers.
Python local Worker children use explicit worker ids rather than deriving
the public worker id from the local worker vector index.

### Responsibilities

- **Pool ownership**: two `std::vector` pools, sized at init from `add_*`
  calls
- **Idle selection & eligibility**: `pick_idle(type, exclude, eligible_worker_ids)`
  finds an idle WorkerThread whose queue is empty, skipping any in `exclude`
  and (when `eligible_worker_ids` is non-empty) restricted to that set; returns
  nullptr if none available. Remote-aware NEXT_LEVEL slots carry final eligible
  worker ids, so a task cannot land on a worker that lacks the callable or
  tensor sidecars.

---

## 2. `WorkerThread`

One WorkerThread per registered mailbox (i.e. per forked child worker).

```cpp
struct WorkerDispatch {
    TaskSlot task_slot;
    int32_t  group_index = 0;    // 0 for non-group; 0..N-1 for group members
};

class WorkerThread {
public:
    void start(Ring *ring, WorkerManager *manager,
               const std::function<void(WorkerCompletion)> &on_complete,
               std::unique_ptr<WorkerEndpoint> endpoint);
    void stop();
    void dispatch(WorkerDispatch d);       // slot id + group sub-index
    bool idle() const;
    const WorkerEndpointCaps &caps() const;
    int32_t worker_id() const;

private:
    Ring *ring_;                       // reads slot state via ring->slot_state(id)
    std::unique_ptr<WorkerEndpoint> endpoint_;
    std::thread thread_;
    std::queue<WorkerDispatch> queue_;
    std::mutex mu_;
    std::condition_variable cv_;

    void loop();
    WorkerCompletion dispatch_process(WorkerDispatch d);
};
```

The WorkerThread's `std::thread` pumps the internal queue and calls
`endpoint->run(...)` once per dispatch. `LocalMailboxEndpoint::run` drives the
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

Each `LocalMailboxEndpoint` drives a `MAILBOX_SIZE`-byte `MAP_SHARED` region.
The Python facade forks one child per mailbox **before**
`WorkerManager::start()` (so the parent has only the Python main thread when
fork runs, avoiding the classical "fork in a multi-threaded process" hazard)
and the child polls the mailbox for the lifetime of the worker.

### 3.1 Parent-side dispatch

```cpp
WorkerCompletion LocalMailboxEndpoint::run(Ring *ring, WorkerDispatch d) {
    TaskSlotState &s = *ring->slot_state(d.task_slot);
    char *m = static_cast<char *>(mailbox_);

    // Write task data: reserved callable field, config, digest prefix, then
    // length-prefixed TaskArgs blob. Tags are stripped; only
    // [digest][T][S][tensors][scalars] crosses the fork boundary.
    uint64_t reserved = 0;
    memcpy(m + MAILBOX_OFF_CALLABLE, &reserved, sizeof(reserved));
    memcpy(m + MAILBOX_OFF_CONFIG, &s.config, sizeof(CallConfig));
    memcpy(m + MAILBOX_OFF_TASK_CALLABLE_HASH, s.callable.digest.data(), 32);
    const TaskArgs &args = s.is_group() ? s.task_args_list[d.group_index] : s.task_args;
    write_blob(m + MAILBOX_OFF_TASK_ARGS_BLOB, args);

    // Signal child
    write_state(mailbox_, MailboxState::TASK_READY);

    // Poll for completion
    while (read_state(mailbox_) != MailboxState::TASK_DONE)
        std::this_thread::sleep_for(std::chrono::microseconds(50));

    int err = read_error(mailbox_);
    write_state(mailbox_, MailboxState::IDLE);
    return err == 0
        ? WorkerCompletion{d.task_slot, d.group_index, EndpointOutcome::SUCCESS, ""}
        : WorkerCompletion{d.task_slot, d.group_index, EndpointOutcome::TASK_FAILURE,
                           read_error_msg(mailbox_)};
}
```

Parent-side cost per dispatch:

- One reserved `uint64`, one `CallConfig`, one 32-byte digest, and one
  TaskArgs blob
- One signal (`write_state`)
- Poll loop with `sleep_for(50us)` (not busy-wait)
- One explicit completion outcome: success, task failure, or endpoint failure

Total ~nanoseconds overhead; the wait is dominated by actual kernel execution.

### 3.2 Child loop

The child loop lives in Python — see `_chip_process_loop` and
`_sub_worker_loop` in `python/simpler/worker.py`. Each child polls
`MAILBOX_OFF_STATE`, decodes the digest-prefixed args blob on `TASK_READY`,
resolves the digest to its private local slot/callable, writes back any error,
and publishes `TASK_DONE`.
The child inherits the parent's full address space at fork time, so:

- ChipCallable objects (pre-fork allocated) are COW-visible at the same VA
- The Python callable registry is COW-visible
- Tensor data in `torch.share_memory_()` regions is fully shared (MAP_SHARED)

### 3.3 Mailbox layout

```text
offset 0:                         int32   state
offset 4:                         int32   error
offset 8:                         uint64  reserved task callable field
                                          or control sub-command
offset 16:                        CallConfig config
MAILBOX_OFF_TASK_CALLABLE_HASH:   uint8[32] callable digest
MAILBOX_OFF_TASK_ARGS_BLOB:       bytes [int32 T][int32 S]
                                        [Tensor x T][uint64_t x S]
tail:                             fixed-size NUL-terminated error message
```

The current mailbox size is the C++ `MAILBOX_SIZE` constant exported through
the nanobind module; Python derives its offsets from the same binding where
possible so the two sides cannot drift silently.

### 3.4 Shutdown

`WorkerManager::shutdown_children()` writes `SHUTDOWN` to every registered
endpoint; for `LocalMailboxEndpoint` this writes `SHUTDOWN` to the mailbox.
Each child loop sees it on its next poll and exits. The Python facade owns the
child PIDs and calls `waitpid()` after writing `SHUTDOWN` to its own mailbox
copy. The parent's `WorkerThread::stop()` only joins the C++ dispatcher thread
— it does not own the child process.

---

## 4. Local vs. Remote Endpoints

The mailbox protocol is the local endpoint contract. Adding another local
forked worker kind still follows the existing pattern:

1. Define the worker entry point.
2. Write a child-process loop that polls the mailbox, decodes the args blob,
   and invokes that entry point.
3. Register the mailbox via `manager.add_next_level_at(worker_id, mailbox)`
   for explicit NEXT_LEVEL worker ids, `manager.add_next_level(mailbox)` for
   the legacy local default, or `manager.add_sub(mailbox)`.

Remote L3 is different. It cannot reuse the mailbox wire format because the
remote side does not share virtual addresses, fork-time COW registries, POSIX
shm names, or parent-visible child PIDs. The remote design introduces a
transport-neutral endpoint under `WorkerThread`: `LocalMailboxEndpoint` wraps
this local mailbox path, while `RemoteL3Endpoint` sends framed TASK, CONTROL,
COMPLETION, HEALTH, and SHUTDOWN messages over the negotiated transport.

The implemented `RemoteL3Endpoint` sends TASK and CONTROL frames, waits for
COMPLETION and CONTROL_REPLY frames through `RemoteL3Transport`, and monitors
an independent simulation health lane. Python remote worker specs open a
session through `simpler-remote-worker`; the endpoint is schedulable only after
the session runner reports `HELLO READY`.

### 4.1 Nested fork ordering (L4+ Worker children)

When an L4 Worker has L3 Worker children, the fork sequence nests:

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

### 5.3 Why one WorkerThread per child?

Alternative: N children share one dispatch queue. Rejected because:

- `WorkerThread` queue is the natural unit of backpressure — if child `i` is
  slow, its queue fills up and scheduler falls back to another
- Simpler mental model: one child = one thread that drives it
- Zero contention on queue access (only one producer, one consumer per queue)

---

## 6. Related

- [hierarchical_level_runtime.md](hierarchical_level_runtime.md) — where this
  layer fits in the three-component engine
- [task-flow.md](task-flow.md) — what `ChipWorker::run` receives
- [scheduler.md](scheduler.md) — the producer of `WorkerThread::dispatch`
  calls
