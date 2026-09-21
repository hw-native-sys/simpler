# Worker Manager — Pool, Progress, and Dispatch

Local task frames carry the submitted callable's 32-byte digest. The child
resolves that digest to its private execution slot; no child-local slot crosses
the mailbox boundary. See
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
[hierarchical-level-runtime.md](hierarchical-level-runtime.md). For what
runs on the other side of the local mailbox, see [task-flow.md](task-flow.md).
For where dispatched tasks come from, see [scheduler.md](scheduler.md).

---

## 1. `WorkerManager`

```cpp
class WorkerManager {
public:
    // Registration (before init). `task_frame_count` selects capacity one or
    // the two-frame endpoint that can stage one successor.
    void add_next_level(void *mailbox, int child_pid = -1,
                        uint32_t task_frame_count = 1);
    void add_next_level_at(int32_t worker_id, void *mailbox,
                           int child_pid = -1,
                           uint32_t task_frame_count = 1);
    void add_next_level_endpoint(std::unique_ptr<WorkerEndpoint> endpoint);
    void add_sub(void *mailbox, int child_pid = -1);

    // Lifecycle
    void start(Ring *ring, OnCompleteFn on_complete,
               OnAcceptFn on_accept);              // creates all endpoint lanes
    void stop_workers();                            // marks lanes stopping
    void stop();

    // Scheduler API
    void progress();
    WorkerThread *get_worker_by_id(WorkerType type, int32_t worker_id) const;
    std::vector<int32_t> next_level_worker_ids() const;
    WorkerThread *pick_idle_sub_excluding(
        const std::vector<WorkerThread *> &exclude) const;

private:
    struct LocalNextLevelEntry {
        int32_t worker_id;
        void *mailbox;
        int child_pid;
        uint32_t task_frame_count;
    };
    struct LocalSubEntry {
        void *mailbox;
        int child_pid;
    };
    std::vector<LocalNextLevelEntry> next_level_entries_;
    std::vector<LocalSubEntry> sub_entries_;
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
- **SUB-only idle selection**: `pick_idle_sub_excluding` chooses an idle SUB
  worker not already used by the same SUB group
- **Scheduler-owned progress**: `progress()` advances every endpoint lane from
  the single Scheduler thread
- **Two-step stop**: `stop_workers()` marks lanes stopping while retaining their
  pool entries; Scheduler progress drains terminal events, then `stop()` clears
  the pools

Callable and remote-buffer eligibility is validated against the exact target
during Orchestrator submission. It is not scheduling metadata and is not
stored on the task slot.

---

## 2. `WorkerThread`

There is one `WorkerThread` endpoint-lane object per endpoint (for a local
endpoint, per forked child), but it does not own a thread or dispatch queue.
The single Scheduler thread submits work and owns activation, acceptance,
completion, and shutdown progress across every lane.

```cpp
struct WorkerDispatch {
    TaskSlot task_slot;
    int32_t  group_index = 0;    // 0 for non-group; 0..N-1 for group members
    uint64_t dispatch_id = 0;    // assigned by WorkerThread on submission
    bool prepare_only = false;   // stage for the FIFO successor
};

class WorkerThread {
public:
    void start(Ring *ring,
               const std::function<void(WorkerCompletion)> &on_complete,
               const std::function<void(WorkerDispatch)> &on_accept,
               std::unique_ptr<WorkerEndpoint> endpoint);
    void stop();
    void dispatch(WorkerDispatch d);          // submit on the active lane
    void dispatch_prepared(WorkerDispatch d); // submit on the staged lane
    bool activate_prepared(RunId run_id);
    void progress();                          // called only by Scheduler
    bool idle() const;                        // active lane is free
    bool can_stage() const;                   // staged lane is free
    bool busy() const;                        // either lane owns work
    const WorkerEndpointCaps &caps() const;
    int32_t worker_id() const;

private:
    Ring *ring_;                       // reads slot state via ring->slot_state(id)
    std::unique_ptr<WorkerEndpoint> endpoint_;
    std::atomic<uint32_t> inflight_;     // endpoint capacity accounting
    std::array<LaneState, 2> lanes_;     // active and staged identities
};
```

Every endpoint is progress-driven: Scheduler submission publishes work,
forwards a latched activation when supported, and polls monotonic
`FRAME_STAGED`, `ACCEPTED`, and `COMPLETED` events. The lane array owns task
identity and disposition while `inflight_` independently enforces endpoint
capacity. The forked child loop lives in Python (`_chip_process_loop`,
`_child_worker_loop`, or `_sub_worker_loop` in `python/simpler/worker.py`); the
parent does not fork children.

`WorkerDispatch` begins with `{task_slot, group_index}`. `WorkerThread` assigns
`dispatch_id` under the lane lock before endpoint submission. A submission
failure terminalizes that same identity exactly once. It also sets
`prepare_only` for the one staged successor. The endpoint combines this carrier
with `slot.run_id`, `slot.pipeline_lease`, `slot.callable`, `slot.task_args`,
and `slot.config` read from `ring->slot_state(task_slot)`. The task frame's
protocol/run/slot/generation/dispatch trailer makes frame reuse and late child
publication detectable.

The two capacity lanes have different meanings:

- The **active lane** accepts ordinary tasks from the FIFO-head run. `idle()`
  refers only to this lane, so an active run cannot use the second frame to run
  two tasks on one device.
- The **staged-successor lane** accepts at most the first eligible single
  NEXT_LEVEL task from the prepared FIFO successor. It remains staged until
  that run becomes the FIFO head and `activate_prepared(run_id)` succeeds.
- Prepared groups dispatch normally after FIFO promotion; they are not split
  across staged lanes.

For a group slot with `group_size() == N`, the Scheduler pushes N ordinary
`WorkerDispatch` entries onto N exact NEXT_LEVEL targets, or N distinct idle
SUB workers. `group_index` selects `task_args_list[i]`. There is no
`WorkerPayload` wrapper.

---

## 3. Dispatch via shm mailbox

Each `LocalMailboxEndpoint` drives a `MAILBOX_SIZE`-byte `MAP_SHARED` region.
The Python facade forks one child per mailbox **before**
`WorkerManager::start()` (so the parent has only the Python main thread when
fork runs, avoiding the classical "fork in a multi-threaded process" hazard)
and the child polls the mailbox for the lifetime of the worker.

The region currently consists of three equal `MAILBOX_FRAME_SIZE` frames:

| Frame | Two-frame chip endpoint | Single-frame endpoint |
| ----- | ----------------------- | --------------------- |
| Base (0) | Startup and control state | Startup and control state |
| Task 0 (1) | Pipeline lease slot 0 | The active task |
| Task 1 (2) | Pipeline lease slot 1 | Unused |

`task_frame_count` is an endpoint contract, not a count inferred from the
allocation size. Parent registration uses each direct chip child's own
published depth, while whole-run admission uses the minimum depth across the
chip set. The current Python facade selects two frames only for a direct A2/A3
onboard chip child whose published pipeline depth is at least two. SUB,
nested-Worker, A5, simulation, and depth-one local endpoints use one task frame
and do not advertise successor staging. Remote endpoints have capacity one and
advance their framed transport incrementally.

### 3.1 Single-frame progress path

`LocalMailboxEndpoint::submit_progress` copies `CallConfig`,
`PipelineSlotLease`, the 32-byte callable digest, and the length-prefixed
`TaskArgs` blob into task frame 0 and publishes `TASK_READY`. The Scheduler
keeps driving the owning endpoint lane and observes acceptance or completion
through `poll_progress`; it does not wait inside one task call. A steady-clock
liveness sample turns a child that exits before completion into
`ENDPOINT_FAILURE`.

The child uses `_run_mailbox_loop`, polls the base control state and task frame
0, resolves the digest to a private local slot, executes the task synchronously,
writes its error tuple, and then publishes `TASK_DONE`. Capacity remains one;
the separate frame prevents a control request from overwriting task state.

### 3.2 Two-frame progress path

A two-frame `LocalMailboxEndpoint` advertises `supports_frame_staging` and is
driven through `submit_progress`, `activate_progress`, and `poll_progress`.
The Scheduler thread is the only parent-side progress owner. The child also
uses one `run_two_frame_loop`; it services the separate control base, stages
both task frames, and owns the bounded active/prepared native lifecycles.

The ordinary active dispatch and the staged successor use distinct initial
states:

```text
active:    IDLE -> TASK_READY    -> FRAME_STAGED -> TASK_LAUNCHED
                                                   -> TASK_DONE | TASK_FAILED
successor: IDLE -> PREPARE_READY -> FRAME_STAGED -> ACTIVATE
                                                   -> TASK_LAUNCHED
                                                   -> TASK_DONE | TASK_FAILED
```

`FRAME_STAGED` means the child validated the frame identity and arguments,
resolved the callable digest, rewrote any mapped host addresses, and retained
an immutable snapshot. The state does not by itself distinguish validation-only
staging from completed native preparation. When both an HBG successor and its
already-active predecessor are non-diagnostic, the child prepares a
generation-bound native token in the successor's leased inactive bank before
publishing `FRAME_STAGED`. A successor that arrives before any active claim
publishes validation-only, allowing the parent to activate it; native prepare
follows that activation or a later predecessor claim without another mailbox
state transition. HBG tasks adjacent to diagnostic state and all TMR tasks keep
the same two-frame protocol but defer native prepare because their shared
diagnostic or device-scratch state cannot be rewritten while another run is
active.

An HBG token remains unlaunched and unaccepted until activation, and no backend
launches a successor until the predecessor is polled and finalized. Shutdown,
stale activation, and pre-launch failure finalize any unlaunched token exactly
once.

Activation is sticky on the parent side: FIFO promotion may be observed before
the child reaches `FRAME_STAGED`. The endpoint records that permission and
publishes `ACTIVATE` only by a compare/exchange from the matching
`FRAME_STAGED` state. The child validates all newly visible frame metadata
before choosing the next active frame by `dispatch_id`; capable native prepare
starts for that frame first, and a successor prepares only after the selected
predecessor owns the active claim. Frame index therefore cannot let a later
dispatch bypass an earlier eligible dispatch.

Task-frame publication briefly shares the base control mutex so it has a
defined order relative to a control request. Once published, ordinary controls
may run while the device task is active. Registry-mutating controls are
deferred until active and backend-prepared native state is finalized; final
unregister also waits for every published frame using that digest to retire.
The default unbounded control wait follows child liveness through this
deferral. A finite timeout includes the deferral interval and poisons the
endpoint if it expires with completion uncertain.

Each task frame is bound to its pipeline lease slot and carries a protocol
trailer with `{run_id, slot_id, generation, dispatch_id}`. Parent and child
recheck that identity at staging, activation, acceptance, and completion. A
stale identity, invalid state, unresolved control timeout, or dead child poisons
the endpoint; the parent returns an endpoint-failure completion for every frame
it still owns instead of reusing uncertain bytes. Those completions are withheld
until the child process has exited (or a non-waitable test endpoint has marked
every owned frame terminal), so releasing a run lease can never race native
cleanup.

### 3.3 Launch acceptance

Launch acceptance is a sticky per-frame word, separate from `MailboxState`.
The parent clears it only immediately before publishing a new task into an
`IDLE` frame. The native launch call receives the word's address and sets it
only when the real launch boundary is crossed. `FRAME_STAGED`, `ACTIVATE`, and
the parent-side progress bookkeeping never manufacture an ACK.

The parent reports `ACCEPTED` at most once per `dispatch_id`, before its terminal
completion when the real word is observed. A task that fails before launch
still advances the run-level acceptance waiter at terminal completion so the
waiter cannot hang, but that conservative fallback does not set the mailbox
word. Endpoint paths with no earlier native marker retain completion-time
acceptance.

The child inherits the parent's full address space at fork time, so:

- ChipCallable objects (pre-fork allocated) are COW-visible at the same VA
- The Python callable registry is COW-visible
- Tensor data in `torch.share_memory_()` regions is fully shared (MAP_SHARED)

### 3.4 Frame layout

```text
frame + 0:                        int32   state
frame + 4:                        int32   error
frame + 8:                        uint64  reserved task field
                                          or base-frame control sub-command
frame + 16:                       CallConfig config / control args
MAILBOX_OFF_PIPELINE_LEASE:       PipelineSlotLease
MAILBOX_OFF_TASK_CALLABLE_HASH:   uint8[32] callable digest
MAILBOX_OFF_TASK_ARGS_BLOB:       bytes [int32 T][int32 S]
                                        [Tensor x T][uint64_t x S]
MAILBOX_OFF_TEARDOWN_REPORT:      SimplerTeardownReport (40 B)
MAILBOX_OFF_SHUTDOWN:             int32 sticky shutdown request
task-frame trailer:               protocol, run id, lease slot,
                                  lease generation, dispatch id
MAILBOX_OFF_ACCEPTED:             int32 sticky native-launch marker
frame tail:                       fixed-size NUL-terminated error message
```

The C++ `MAILBOX_FRAME_SIZE` and `MAILBOX_SIZE` constants are exported through
the nanobind module. Python derives frame slicing and offsets from those
bindings where possible. `MAILBOX_ARGS_CAPACITY` ends before the teardown
record, shutdown word, protocol trailer, acceptance word, and error-message
tail.

`MAILBOX_OFF_SHUTDOWN` carries the termination request on the control base
frame. The state word has three writers — the parent's `CONTROL_REQUEST`, the
child's `CONTROL_DONE`, and the endpoint's return to `IDLE` — so a `SHUTDOWN`
state store is erasable. Only a terminating parent writes the shutdown word,
0 -> 1, and nothing clears it, so a child's serve loop still observes the
request after an in-flight control command completes over it. Both sides write
it before the `SHUTDOWN` state store and read it at the top of every serve-loop
iteration alongside the state word.

`MAILBOX_OFF_TEARDOWN_REPORT` carries the chip child's observation of its own
device teardown, published once on the control base frame after the child's
`ChipWorker::finalize()` returns or raises. It is a record rather than a state
word because its fields only mean anything together: which reset entry ran,
what that call returned, what the recovery wrapper and its post-reset probe
returned, and what the teardown returned, each kept separate so a single error
code cannot conflate "no reset ran", "the reset failed" and "the reset
returned 0 but nothing confirmed it". `schema` at offset 0 is the commit
marker — the child fills the payload first and releases `schema` last, so a
parent that acquire-loads it has the whole record and a parent that finds
anything else has none of it. The parent copies it out in its child-reap loop,
after the child is reaped and before the shared memory is closed, and exposes
it through `Worker.teardown_reports()`.

It is observation only: no field asserts that device work has stopped, and a
confirmed reset invalidates that device generation's allocations rather than
making an old device pointer reusable. The onboard platform runner is shared
by every runtime built on it, so publishing is gated on the runtime's own
`teardown_report_supported_impl` — a2a3 `host_build_graph` opts in, and every
other runtime answers "no observation", which reads back as an uncommitted
record. Remote endpoints have no local mailbox and produce no entry at all.

A chip child that outlives the close deadline is handed to the cleanup
journal, and its record is read there instead: inside the same retry closure,
after the reap succeeds and before the shm is closed. So a late publication is
still collected, and a missing key keeps meaning "never reaped".

### 3.5 Stop and child shutdown

`Worker::close()` first asks the Scheduler to stop admitting dispatch, then
calls `WorkerManager::stop_workers()` while Scheduler callbacks and worker pool
entries are still valid. A `WorkerThread` repeatedly publishes
`SHUTDOWN` on the control base until its frames terminalize; the sticky
shutdown word (section 3.4) is what makes the request itself unloseable, while
the repetition keeps the *state* word from resting on a `CONTROL_DONE` a
concurrently finishing control handler published. The child finalizes any
active native run and marks every
active or staged task frame failed before leaving; the parent drains those
terminal events, brings its in-flight count to zero, and joins the one progress
thread. The Scheduler is joined only after worker threads can no longer invoke
completion callbacks, and `WorkerManager::stop()` then clears the pools.

A single-frame endpoint has no staged frame to cancel; its active task
terminalizes before its dispatcher thread joins. The Python facade owns every
child PID. After native worker teardown it broadcasts `SHUTDOWN` to all child
base frames, reaps them under one shared deadline, and closes their shared-memory
objects. C++ endpoint code may observe or reap an early child exit for liveness
diagnostics, but it does not own the normal `waitpid()` lifecycle.

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

The implemented `RemoteL3Endpoint` incrementally sends TASK frames and polls
COMPLETION frames through `RemoteL3Transport`. CONTROL/CONTROL_REPLY exchanges
remain serialized on the same ordered command lane, and an independent
simulation health lane is monitored throughout. Python remote worker specs open
a session through `simpler-remote-worker`; the endpoint is schedulable only
after the session runner reports `HELLO READY`.

A control request issued while a remote task is in flight waits for that task
to complete or for endpoint progress to stop. The command and task share one
ordered socket lane, so the remaining task runtime is part of the control
request's worst-case latency.

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

### 5.3 Why one progress owner for every child?

Alternative: one thread per child, each driving its own endpoint. Rejected
because `submit_progress` / `poll_progress` never wait for child completion, so
a dedicated thread has no completion to block on — it spins for as long as its
child is busy, and N children cost N spinning cores. The Scheduler already wakes
on completions and already owns the dispatch decision, so folding progress into
it costs one poll round per iteration and removes both the threads and the queue
that fed them.

Not waiting for completion is not the same as never blocking: `submit_progress`
still takes `mailbox_mu_`, and the second bullet below is what that costs.

What that trades away:

- One endpoint's progress latency is now bounded by the whole round, since a
  single thread visits every lane in turn rather than each lane having its own
- A blocking call reached from the progress path stalls every lane, not just
  its own. `submit_progress` waits on `mailbox_mu_`, which `run_control_command`
  holds while it blocks on the child; see the comment at that acquisition in
  `worker_manager.cpp`

What is unchanged: per-child work still queues per child. Directed NEXT_LEVEL
work waits in child `i`'s ready FIFO if that child is busy, while SUB work may
use another idle SUB child.

---

## 6. Related

- [hierarchical-level-runtime.md](hierarchical-level-runtime.md) — where this
  layer fits in the three-component engine
- [task-flow.md](task-flow.md) — what `ChipWorker::run` receives
- [scheduler.md](scheduler.md) — the producer of `WorkerThread::dispatch`
  calls
