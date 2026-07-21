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

`WorkerEndpointCaps::max_in_flight` defines the dispatch credits of one
endpoint. `WorkerThread` owns that many executor threads and reports itself
available while at least one credit remains. Onboard a2a3 HostGraph chip
mailboxes advertise two credits; TRB, simulation, SUB, nested Worker, and
remote endpoints advertise one.

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
    WorkerThread *get_worker_by_id(WorkerType type, int32_t worker_id) const;
    std::vector<int32_t> next_level_worker_ids() const;
    WorkerThread *pick_idle_sub_excluding(
        const std::vector<WorkerThread *> &exclude) const;

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
- **Directed NEXT_LEVEL lookup**: `get_worker_by_id` resolves the exact stable
  target selected by the user; the Scheduler never asks the manager to choose
  another NEXT_LEVEL worker
- **SUB-only credit selection**: `pick_idle_sub_excluding` chooses a SUB worker
  with available endpoint credit that is not already used by the same SUB group

Callable and remote-buffer eligibility is validated against the exact target
during Orchestrator submission. It is not scheduling metadata and is not
stored on the task slot.

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
    std::vector<std::thread> threads_;
    std::queue<WorkerDispatch> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
    uint32_t capacity_;
    std::atomic<uint32_t> available_;

    void loop();
    WorkerCompletion dispatch_process(WorkerDispatch d);
};
```

The WorkerThread executor threads pump one shared queue and call
`endpoint->run(...)` once per reserved credit. `LocalMailboxEndpoint::run`
drives one mailbox frame round trip per dispatch. The forked child loop
that consumes the mailbox lives in Python (`_chip_process_loop` /
`_sub_worker_loop` in `python/simpler/worker.py`); the parent does not fork
children.

`WorkerDispatch` carries only `{slot_id, group_index}`; the thread reads
`slot.callable` / `slot.task_args` / `slot.config` on each dispatch via
`ring->slot_state(slot_id)`. For a group slot with `group_size() == N`,
the Scheduler pushes N `WorkerDispatch` entries (one per member) onto N
exact target threads for NEXT_LEVEL, or N freely selected SUB threads. Each
thread's `group_index` selects which
`task_args_list[i]` view to hand to the worker. There is no
`WorkerPayload` — the per-dispatch carrier is just the slot id plus the
group sub-index.

---

## 3. Dispatch via shm mailbox

Each `LocalMailboxEndpoint` drives a `MAILBOX_SIZE`-byte `MAP_SHARED` region
containing two fixed-size task frames. Slot generation identifies frame reuse;
the protocol magic/version rejects mismatched parent and child layouts. An
onboard a2a3 HostGraph child owns two long-lived task threads. Each thread has
its own `ChipWorker`/`DeviceRunner`, Runtime storage, streams, kernel arguments,
diagnostic collectors, and private device arena. This isolates Host run state
instead of making `DeviceRunnerBase`'s mutable launch state concurrent.
The Python facade forks one child per mailbox **before**
`WorkerManager::start()` (so the parent has only the Python main thread when
fork runs, avoiding the classical "fork in a multi-threaded process" hazard)
and the child polls the mailbox for the lifetime of the worker.

### 3.1 Parent-side dispatch

```cpp
WorkerCompletion LocalMailboxEndpoint::run(Ring *ring, WorkerDispatch d) {
    TaskSlotState &s = *ring->slot_state(d.task_slot);
    size_t slot = acquire_task_slot();
    char *m = static_cast<char *>(mailbox_) + slot * MAILBOX_TASK_SLOT_SIZE;

    // Write task data: generation, protocol, config, digest prefix, then
    // length-prefixed TaskArgs blob. Tags are stripped; only
    // [digest][T][S][tensors][scalars] crosses the fork boundary.
    uint64_t generation = next_generation(slot);
    memcpy(m + MAILBOX_OFF_GENERATION, &generation, sizeof(generation));
    write_protocol(m, MAILBOX_PROTOCOL_MAGIC_VERSION);
    memcpy(m + MAILBOX_OFF_CONFIG, &s.config, sizeof(CallConfig));
    memcpy(m + MAILBOX_OFF_TASK_CALLABLE_HASH, s.callable.digest.data(), 32);
    const TaskArgs &args = s.is_group() ? s.task_args_list[d.group_index] : s.task_args;
    write_blob(m + MAILBOX_OFF_TASK_ARGS_BLOB, args);

    // Signal child
    write_state(m, MailboxState::TASK_READY);

    // Poll for completion
    while (read_state(m) != MailboxState::TASK_DONE)
        std::this_thread::sleep_for(std::chrono::microseconds(50));

    check_generation(m, generation);
    int err = read_error(m);
    write_state(m, MailboxState::IDLE);
    release_task_slot(slot);
    return err == 0
        ? WorkerCompletion{d.task_slot, d.group_index, EndpointOutcome::SUCCESS, ""}
        : WorkerCompletion{d.task_slot, d.group_index, EndpointOutcome::TASK_FAILURE,
                           read_error_msg(m)};
}
```

Parent-side cost per dispatch:

- One generation `uint64`, one protocol word, one `CallConfig`, one 32-byte
  digest, and one
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

Control commands remain exclusive with task frames. Request-session resources
that require control operations, such as L3-L2 message regions, must therefore
be created before the first device task is published. The two-flight hardware
test creates both token queues before request A submits its task; request B then
reuses its existing queue without draining A.

### 3.3 Mailbox layout

```text
offset 0:                         int32   state
offset 4:                         int32   error
offset 8:                         uint64  task generation
                                          or control sub-command
offset 16:                        CallConfig config
MAILBOX_OFF_TASK_CALLABLE_HASH:   uint8[32] callable digest
MAILBOX_OFF_TASK_ARGS_BLOB:       bytes [int32 T][int32 S]
                                        [Tensor x T][uint64_t x S]
MAILBOX_OFF_PROTOCOL:             uint64  protocol magic/version
frame tail:                       fixed-size NUL-terminated error message
```

`MAILBOX_SIZE` is two `MAILBOX_TASK_SLOT_SIZE` frames. The constants are
exported through the nanobind module; Python derives its protocol, offsets,
and allocation size from that binding so the two sides cannot drift silently.

### 3.4 Shutdown

`WorkerManager::shutdown_children()` writes `SHUTDOWN` to every registered
endpoint; for `LocalMailboxEndpoint` this writes `SHUTDOWN` to the mailbox.
Each child loop sees it on its next poll and exits. The Python facade owns the
child PIDs and calls `waitpid()` after writing `SHUTDOWN` to its own mailbox
copy. The parent's `WorkerThread::stop()` only joins the C++ dispatcher threads
— it does not own the child process.

---

## 4. Local vs. Remote Endpoints

The mailbox protocol is the local endpoint contract. Adding another local
forked worker kind still follows the existing pattern:

1. Define the worker entry point.
2. Write a child-process loop that polls the mailbox, decodes the args blob,
   and invokes that entry point.
3. Register the mailbox via `manager.add_next_level_at(worker_id, mailbox)`
   for an explicit NEXT_LEVEL worker id, `manager.add_next_level(mailbox)` to
   allocate the next stable local id, or `manager.add_sub(mailbox)`.

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

When an L4 Worker has L3 Worker children, `init()` is eager and the fork
sequence nests recursively — the whole tree is READY when `L4.init()` returns:

```text
L4 parent process
  ├─ _init_hierarchical(): Worker(4) + HeapRing mmap (before fork)
  └─ _start_hierarchical() (inside init()):
       ├─ fork L3 child  ────────►  L3 child process:
       │                              inner_worker.init()  ← eager, recursive
       │                                ├─ Worker(3) + L3 HeapRing
       │                                └─ _start_hierarchical() forks L3's
       │                                   sub/chip children and blocks on their
       │                                   INIT_READY, THEN publishes INIT_READY
       │                              _child_worker_loop()  (dispatch only)
       ├─ await every L3 child's INIT_READY (whole subtree ready)
       └─ register mailbox with L4's Worker
```

Each inner Worker inits **inside its forked child process** so its own
children are forked from the correct parent. Because the inner `init()` is
eager and blocks on its descendants, a child publishes `INIT_READY` only after
its whole subtree is ready, so readiness propagates recursively up to
`L4.init()`. The L4 parent never sees L3's sub/chip grandchildren — they're
L3's responsibility; if startup fails, they are reclaimed via the child's
process-group cancellation domain, not the resource_tracker.

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

- `WorkerThread` is the natural execution unit. Directed NEXT_LEVEL work waits
  in child `i`'s ready FIFO if that child is busy; SUB work may use another
  idle SUB child
- Simpler mental model: one child = one thread that drives it
- Zero contention on queue access (only one producer, one consumer per queue)

---

## 6. Related

- [hierarchical_level_runtime.md](hierarchical_level_runtime.md) — where this
  layer fits in the three-component engine
- [task-flow.md](task-flow.md) — what `ChipWorker::run` receives
- [scheduler.md](scheduler.md) — the producer of `WorkerThread::dispatch`
  calls
