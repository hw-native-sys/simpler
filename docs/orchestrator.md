# Orchestrator — DAG Submission Internals

Callable identity update: the Python facade validates `CallableHandle` objects
and passes callable digest/kind/namespace metadata into the C++ Orchestrator.
Older `callable_id`/`cid` examples below are target-local or historical
internals, not public submit arguments. See
[callable-identity-registration.md](callable-identity-registration.md).

The Orchestrator is the **DAG builder**. It runs single-threaded on the user's
thread while a run is open for submission and owns the three data structures
that turn a sequence of `submit_*` calls into a scheduled DAG: `Ring`,
`TensorMap`, and `Scope`.

For the high-level role of the Orchestrator among the three engine components,
see [hierarchical-level-runtime.md](hierarchical-level-runtime.md). For what
flows through `submit`, see [task-flow.md](task-flow.md).

---

## 1. Python Facade and C++ Internal API

The Python user's orch fn receives a `simpler.orchestrator.Orchestrator`
facade. `submit_next_level` and `submit_next_level_group` enqueue DAG nodes and
return opaque `TaskHandle` objects. A later `TaskArgs.add_dep(handle)` or
`TaskArgs.add_dep_wait(handle)` uses the handle to add an edge that tensor-tag
inference cannot derive.
`submit_sub` and `submit_sub_group` still return `None`.

The C++ `SubmitResult` name is an alias of the same run-scoped handle:

```cpp
class Orchestrator {
public:
    // --- Internal submit API (tags inside TaskArgs drive deps) ---
    SubmitResult submit_next_level(const CallableIdentity &callable,
                                    const TaskArgs &args,
                                    const CallConfig &config,
                                    int32_t worker,
                                    const std::vector<int32_t> &eligible_worker_ids = {},
                                    const RemoteTaskArgsSidecar &remote_sidecar = {});
    SubmitResult submit_next_level_group(const CallableIdentity &callable,
                                          const std::vector<TaskArgs> &args_list,
                                          const CallConfig &config,
                                          const std::vector<int32_t> &workers,
                                          const std::vector<std::vector<int32_t>> &eligible_worker_ids = {},
                                          const std::vector<RemoteTaskArgsSidecar> &remote_sidecars = {});
    SubmitResult submit_sub(const CallableIdentity &callable,
                            const TaskArgs &args);
    SubmitResult submit_sub_group(const CallableIdentity &callable,
                                   const std::vector<TaskArgs> &args_list);

    // --- Intermediate-buffer allocation (runtime-owned lifetime) ---
    uint64_t alloc(const std::vector<uint32_t> &shape, DataType dtype,
                   const CanonicalIdentity &identity);

    // --- Internal lifecycle (invoked by Python Worker.submit/RunHandle) ---
    RunId begin_run();
    void close_run_submission(RunId run_id);
    void fail_run_submission(RunId run_id, std::exception_ptr error);
    void wait_run_accepted(RunId run_id);
    bool run_accepted(RunId run_id) const;
    void wait_run(RunId run_id);
    bool wait_run_for(RunId run_id, double timeout_seconds);
    bool run_done(RunId run_id) const;
    void release_run(RunId run_id);
    void scope_begin();
    void scope_end();

private:
    // ... components: Ring, TensorMap, Scope, and the RunState registry
};

struct TaskHandle {
    RunId run_id;
    TaskSlot task_slot;
};
using SubmitResult = TaskHandle;
```

Python does not expose either field or a public constructor. A handle is valid
only while building the run that returned it; passing it to another run is a
submit-time error. Naming a producer that has already completed or reached
`CONSUMED` in the same run is valid and the dependency is already satisfied.
The monotonic `run_id` prevents a stale handle from accidentally naming a slot
reused after the Ring resets.

**Status**: `submit_sub` takes only `(CallableIdentity, args)` — no
`config`, since SUB has no per-call config.

The run lifecycle and outer `scope_begin` / `scope_end` are invoked from Python
`Worker.submit` through private bindings. They are not part of the user-facing
orch-fn API. `Worker.submit` invokes the orchestration callback and closes DAG
submission synchronously, then returns a `RunHandle` before L3 device work has
necessarily completed. Graph-construction errors therefore remain synchronous;
device or endpoint errors are attached to the handle and raised by `wait()` or
`result()`.

`RunHandle.done` polls the matching native run fence. `wait(timeout)` supports
bounded waits without cancelling or corrupting the run, and repeated waits
replay the same terminal result. The handle keeps its `Worker`, callback
arguments, configuration, and run-owned cleanup state alive until completion.
`Worker.close()` rejects new submissions and drains operation leases plus
accepted handles within one cleanup budget before tearing down the worker
tree. If that budget expires, teardown remains unattempted and the closed
worker retains the handles/tree for a later `close()` retry.

`Worker.run` remains source-compatible and blocking:

```python
worker.run(orchestration, args, config)
# Equivalent to:
worker.submit(orchestration, args, config).wait()
```

L2 `submit()` returns a `RunHandle` for the chip run admitted by its native
lane. The backend's negotiated capability determines whether a successor may
prepare early or submission must drain its predecessor. Successful `wait()` /
`result()` completes the run and its required finalization. At L3 and above, graph
callbacks remain serialized. A later `submit` is admitted while logical
pending-run capacity remains available; the default capacity follows the
negotiated depth. `begin_run` may construct such a pending run without a native
lease, and `refresh_leases_locked` assigns the lease when the run enters an
active or preparable role. Endpoint acceptance remains the launch fence a run's
own dispatches advance — on A2A3 onboard, after both device kernels are
enqueued and before stream synchronization, with endpoints lacking an earlier
signal falling back to completion — but it no longer gates the next callback.
Each run still owns its completion error, keepalives, and cleanup independently.

`close_run_submission()` closes task declarations; tasks may still be PENDING
or READY. A second `begin_run()` is rejected while construction remains open,
but the executing FIFO head can dispatch tasks before its callback returns.
An eligible successor may be sent to its endpoint with `prepare_only` while
the predecessor executes. Closing submission or accepting every predecessor
task does not activate that successor: the current FIFO advances only when
the predecessor becomes terminal. A successor promoted while its callback is
still open may then execute its ready tasks.

Python direct-control calls inside a callback must precede its first task
submission and wait for that run to hold the FIFO head. Ownerless control
requires the Worker's control reservation and quiescence. Device-touching
run cleanup additionally retains its ordered-cleanup barrier until published.

Remote L3 submit adds two hidden pieces of metadata: final eligible worker-id
sets and optional `RemoteTaskArgsSidecar` entries aligned by tensor index.
Python `RemoteCallable` handles supply callable eligibility, and
`TaskArgs.add_tensor(RemoteTensorRef(...), tag)` supplies tensor sidecars. The
Orchestrator validates exact placement, worker existence, local-vs-remote
compatibility, remote handle access rights for the tensor tag, bare host
pointers, and remote null OUTPUT tensors before committing the slot.
For NEXT_LEVEL tasks, `worker`/`workers` are required stable worker ids rather
than C++ worker-thread vector indices. SUB submit APIs expose no worker
selection and use the shared SUB ready queue.

---

## 2. `submit_next_level` — the 7-step flow

This is the entry point for every task in the DAG. All submit variants share
the same skeleton; `submit_next_level_group` and `submit_sub` differ only in
how the slot is set up.

```cpp
SubmitResult Orchestrator::submit_next_level(const CallableIdentity &callable,
                                              TaskArgs args,
                                              const CallConfig &config,
                                              int32_t worker) {
    // 1. Alloc slot (blocks on back-pressure if ring full)
    TaskSlot sid = ring_.alloc();
    TaskSlotState &s = slots_[sid];
    s.reset();

    // 2. Move task data into slot (parent heap, no encoding)
    s.worker_type = WorkerType::NEXT_LEVEL;
    s.callable    = callable;
    s.task_args   = std::move(args);
    s.config      = config;
    s.target_worker_ids = {worker};

    // 3. Collect explicit deps, then walk task_args tags to derive retained
    //    dependencies. Repeated producers collapse to one edge and retention
    //    wins when any path names the same producer with retain=true.
    std::vector<ProducerDependency> producers;
    for (int i = 0; i < s.task_args.explicit_dep_count(); ++i) {
        add_unique_producer(s.task_args.explicit_dep(i).task_slot,
                            s.task_args.explicit_dep_retain(i));
    }
    for (int i = 0; i < s.task_args.tensor_count(); i++) {
        TensorArgType tag = s.task_args.tag(i);
        TensorKey key     = key_for_tensor_or_remote_sidecar(i);

        if (tag == INPUT || tag == INOUT) {
            if (TaskSlot prod = tensormap_.lookup(key); prod != INVALID)
                add_unique_producer(prod, /*retain=*/true);
        }
        if (tag == OUTPUT || tag == INOUT || tag == OUTPUT_EXISTING) {
            tensormap_.insert(key, sid);
        }
        // NO_DEP: skip both
    }

    // 4. Register with scope (holds slot open until scope_end releases ref)
    scope_.register_task(sid);
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        s.fanout_total += 1;
    }

    // 5. Attach fanout edges under each producer's mutex. Producers already
    //    completed do not count as live fanins; failed producers poison this
    //    slot.
    int32_t live = attach_fanout_and_count_live_producers(sid, producers);

    // 6. Publication: the one point where the slot leaves BUILDING. The count
    //    and the transition are published together under fanout_mu — see
    //    §8 "The BUILDING publication rule".
    bool ready;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        s.fanin_count.store(live);
        ready = s.fanin_released >= live;   // releases that landed while BUILDING
        if (!s.state.compare_exchange_strong(BUILDING, ready ? READY : PENDING)) {
            propagate_failure(sid);         // a producer claimed us mid-wiring
            return {run_id, sid};
        }
    }
    if (ready) enqueue_ready(sid);          // outside the lock: takes runs_mu_

    // 7. Return handle
    return {run_id, sid};
}
```

### Step details

**Step 1 — `ring_.alloc()`**: See [§5 Ring](#5-ring-slot-and-per-scope-heap-allocator). Blocks the Orch thread
if all slots are in-flight; this is the system's back-pressure mechanism.

**Step 2 — store task data**: `TaskArgs` is moved (not copied). `config` is a
small POD copied by value. `callable` is a `uint64_t` opaque handle (see
[task-flow.md](task-flow.md) §2).

**Step 3 — explicit deps and tag walk**: The only place dependency handles and
tags are consumed. Neither is carried across the dispatch boundary (see
[task-flow.md](task-flow.md) §3).
Every TensorMap key starts with the current `RunId`. Local tensor identity is
then `(LOCAL_HOST, ptr)` or `(LOCAL_CHILD, worker, ptr)`. Remote tensors with
sidecars use `(address_kind, owner_worker_id, buffer_id, generation, offset)`.

| Tag | `tensormap.lookup` | `tensormap.insert` |
| --- | ------------------ | ------------------ |
| `INPUT` | ✓ | — |
| `OUTPUT` | — | ✓ |
| `INOUT` | ✓ | ✓ |
| `OUTPUT_EXISTING` | — | ✓ |
| `NO_DEP` | — | — |

`OUTPUT_EXISTING` differs from `OUTPUT` in runtime semantics (user-provided
buffer vs. runtime-allocated) but dependency tracking is identical: both
register this task as the new producer of the tensor's dependency key. For
local tensors this key contains `tensor.data`; for remote sidecars it contains
remote buffer identity and logical offset.

`TaskArgs.add_dep(*handles)` and `TaskArgs.add_dep_wait(*handles)` accept one or
more handles and may be called before or after `add_tensor` / `add_scalar`.
Their dynamic host-side vector has no fixed dependency cap. Every handle must
belong to the current run and name an already submitted task. Both edge kinds
contribute to the consumer's `fanin_count`, propagate producer failure, and
delay READY until the producer finishes. `add_dep` additionally retains each
producer until the consumer finishes; `add_dep_wait` is ordering-only. Neither
creates a TensorMap entry. Repeated handles and tag-inferred edges deduplicate,
with lifetime retention winning if any path requests it.

Before each output mapping becomes visible, its key is appended to the slot's
cleanup journal. A failed journal append therefore publishes no mapping, while
a later failure leaves a key `on_consumed` can erase. Erasure remains
owner-checked: if publication failed before replacing an older producer, cleanup
of the failed slot does not remove that older producer's mapping.

**Step 4 — scope ref**: A slot submitted inside an open scope registers one
scope reference in `fanout_total`. Registration precedes the charge, so a
failed registration cannot leave a reference that no `scope_end` can release.
See [§6 Scope](#6-scope).

**Step 5 — fanout attachment and live-fanin count**: Submission synchronously
locks each producer's `fanout_mu`, attaches the consumer, and counts only live
producers. Every edge registers the consumer for completion notification.
Tensor-derived and explicit `add_dep` edges additionally charge `fanout_total`
and record a reverse edge so the consumer retains the producer until it
finishes; `add_dep_wait` edges omit both lifetime operations. A completed
producer contributes no live fanin. A completed wait-only dependency needs no
notification edge at all.

**Step 6 — publication and READY routing**: `fanin_count` and the transition
from BUILDING to PENDING or READY are published together under `fanout_mu`. An
immediately READY task is routed to its exact NEXT_LEVEL worker FIFO, the
NEXT_LEVEL group FIFO, or the shared SUB FIFO. The same routing function is
used when Scheduler releases the final dependency. See
[scheduler.md](scheduler.md) §1.

---

## 3. `submit_next_level_group` — N workers, 1 DAG node

A group task is a single DAG node that executes in parallel on N workers.
Each worker gets its own `TaskArgs`; the node only reaches COMPLETED when all
N finish.

Callers submit tasks that wait for same-level peers as one complete group.
Submitting those members as independent singles can start one member before
its peers are READY, leaving the running member unable to finish.

```cpp
SubmitResult Orchestrator::submit_next_level_group(
    const CallableIdentity &callable, const std::vector<TaskArgs> &args_list,
    const CallConfig &config, const std::vector<int32_t> &worker_ids,
    const std::vector<std::vector<int32_t>> &eligible_worker_ids,
    const std::vector<RemoteTaskArgsSidecar> &remote_sidecars
) {
    return submit_impl(
        WorkerType::NEXT_LEVEL, callable, config, args_list, worker_ids,
        eligible_worker_ids, remote_sidecars
    );
}
```

`submit_impl` validates that `worker_ids` contains one unique, eligible target
per group member before it performs shared dependency inference and READY
routing. It also prepares the member-state and member-outcome vectors as one
transaction while the slot is still BUILDING. Dispatch therefore never has to
allocate after claiming READY → RUNNING. A defensive size repair is likewise
prepared in local vectors before the claim and published only if the Scheduler
wins; a losing claim cannot overwrite cancellation's terminal bookkeeping.

At dispatch time the Scheduler checks the group FIFO head and resolves every
entry in `workers` to that exact stable worker ID. It dispatches only if the
entire target set is idle. A blocked group reserves all of its targets against
new singles but does not cause a scan past the FIFO head. Each member lane is
submitted with its own `task_args_list[i]`. Completion remains
aggregated at the group slot, so downstream consumers are released once after
every member is terminal. Completion validates both bookkeeping-vector sizes;
repair preserves already-terminal and still-running members before indexing.

---

## 4. `submit_sub` — Python callable leaf

Sub tasks resolve a Python function by callable digest in the SUB child:

```cpp
SubmitResult Orchestrator::submit_sub(const CallableIdentity &callable, TaskArgs args) {
    TaskSlot sid = ring_.alloc();
    TaskSlotState &s = slots_[sid];
    s.reset();
    s.worker_type = WorkerType::SUB;
    s.callable    = callable;
    s.task_args   = std::move(args);

    std::vector<TaskSlot> producers;
    std::unordered_set<TaskSlot> producers_seen;
    for (int i = 0; i < s.task_args.tensor_count(); i++) {
        TensorArgType tag = s.task_args.tag(i);
        uint64_t ptr      = s.task_args.tensor(i).data;
        if (tag == INPUT || tag == INOUT)
            if (auto prod = tensormap_.lookup(ptr); prod != INVALID)
                if (producers_seen.insert(prod).second)
                    producers.push_back(prod);
        if (tag == OUTPUT || tag == INOUT || tag == OUTPUT_EXISTING)
            tensormap_.insert(ptr, sid);
    }

    s.fanin_count = static_cast<int32_t>(producers.size());
    scope_.register_task(sid);
    scheduler_.enqueue_wiring(sid, std::move(producers));
    return {sid};
}
```

---

## 5. Ring (slot and per-scope heap allocator)

`Ring` owns three correlated per-task resources:

1. A **monotonic task id** — allocated on every `alloc()`, shared across
   all rings. There is no fixed window and no modulo wrap at L3: slot
   state lives in parent-process heap (never crossed into child workers),
   so the ring-index addressing scheme L2 needs for shmem descriptors
   buys us nothing here. A monotonic `int32_t` gives ~2 billion ids per
   globally quiescent compaction interval.
2. **`MAX_RING_DEPTH = 4` independent shared-memory heap slabs**
   (Strict-1; matches L2's `CHIP_MAX_RING_DEPTH`). Each slab has its own
   `mmap(MAP_SHARED | MAP_ANONYMOUS)` region, bump cursor, FIFO
   reclamation pointer, and mutex / cv. A task's ring is chosen by
   **scope depth**:

   ```cpp
   ring_idx = std::min(scope_depth, MAX_RING_DEPTH - 1);
   ```

   so nested-scope tasks never share a FIFO head with outer-scope
   long-lived allocations. The mapping is taken in the Worker ctor —
   *before* any fork — so forked child workers inherit every ring at
   the same virtual address range. `heap_ring_size` on the Worker ctor
   is the **per-ring** size (default 1 GiB → 4 GiB total VA reservation;
   physical pages stay lazy).
3. The **per-task slot state** (`TaskSlotState`) — stored in a single
   `std::deque<std::unique_ptr<...>>` shared across rings. Each slot
   records its `ring_idx` and `ring_slot_idx` (position within that
   ring's FIFO order). `std::deque::push_back` never invalidates pointers
   to existing elements, so the pointer returned by `slot_state(id)`
   stays valid until globally quiescent `reset_to_empty()` drops the deque.

```cpp
struct AllocResult {
    TaskSlot slot;
    void    *heap_ptr;          // nullptr when alloc(0)
    uint64_t heap_end_offset;   // byte offset within the selected ring
    int32_t  ring_idx;          // which of the MAX_RING_DEPTH rings was used
};

class Ring {
public:
    // Initialise MAX_RING_DEPTH heap rings, each of heap_bytes.
    // Total VA = MAX_RING_DEPTH * heap_bytes.
    void init(uint64_t heap_bytes,       // per-ring, default 1 GiB
              uint32_t timeout_ms);      // default 10 s

    // Pick ring = min(scope_depth, MAX_RING_DEPTH - 1); reserve a
    // slab from that ring (blocks on its cv) and stamp the slot state
    // with ring_idx / ring_slot_idx.
    AllocResult alloc(uint64_t bytes = 0, int32_t scope_depth = 0);
    void            release(TaskSlot sid);      // FIFO-advances THAT slot's ring
    TaskSlotState *slot_state(TaskSlot sid);
    void            reset_to_empty();           // rewinds every ring
    void            shutdown();

    // Per-ring introspection
    void    *heap_base(int32_t ring_idx) const;
    uint64_t heap_size(int32_t ring_idx) const;
    uint64_t heap_top (int32_t ring_idx) const;
    uint64_t heap_tail(int32_t ring_idx) const;
};
```

**Back-pressure is per-ring**: only the selected ring's heap can be full
for a given `alloc()`. `alloc()` spin-waits on that ring's cv while
other rings remain fully usable; if `timeout_ms` elapses with no
progress, it throws `std::runtime_error`. That surfaces as a Python
exception so users can enlarge `heap_ring_size` on the `Worker` instead
of deadlocking.

**Alignment**: every heap allocation is rounded up to `HEAP_ALIGN = 1024 B`
(matches L2's `PACKED_OUTPUT_ALIGN`, Strict-3).

**FIFO reclamation per ring**: each `alloc()` appends the slot's
`heap_end_offset` onto the selected ring's `slot_heap_end[]` vector, and
pushes a `released=0` byte. `release(slot)` looks up the slot's ring via
`slot.ring_idx` and advances **that ring's** `last_alive` as long as the
next-oldest in-ring slot is released, walking the ring's `heap_tail`
forward. Rings never touch each other — inner-scope tasks reclaim
without waiting for an outer-scope task to finish.

**Run completion and compaction**: every committed slot increments its owning
`RunState.active_tasks`; `on_consumed` decrements that same run exactly once. A
run becomes terminal only after submission is closed and its count reaches
zero. Slot and heap reclamation still happens incrementally through
`ring.release(slot)`. `release_run()` may call `ring.reset_to_empty()` only
when no registered runs or live slots remain, so one run never resets storage
owned by another.

**Locking**: each ring has its own `mu` / `cv`; the shared
`next_task_id_` and slot deque are guarded by a separate `slots_mu_`.
`alloc()` holds ring.mu (back-pressure wait + reserve in-ring position),
releases it, then takes `slots_mu_` briefly to publish the new slot —
no nested locking. `reset_to_empty()` takes `slots_mu_` first and each
ring's mu sequentially (nested, outer is `slots_mu_`); readers that
need both lock in the same order.

**Ownership by role**:

| Field | Writer | Reader |
| ----- | ------ | ------ |
| `next_task_id_`, `slot_states_` | `alloc` under `slots_mu_` | `slot_state`, `next_task_id`, `reset_to_empty` |
| `rings_[r].top`, `rings_[r].released[]`, `rings_[r].slot_heap_end[]` | `alloc` under `rings_[r].mu` | `release` under `rings_[r].mu`, introspection accessors |
| `rings_[r].tail`, `rings_[r].last_alive` | `release` under `rings_[r].mu` | same; `reset_to_empty` |
| `slot.ring_idx`, `slot.ring_slot_idx` | `alloc` (stamped before return) | `release` |

---

## 6. Scope

Scope solves two concerns at once:

1. **Lifetime anchoring** — release a task's ring slot even when it has no
   downstream consumer, so leaf tasks don't strand heap bytes.
2. **Per-scope reclamation** — tasks submitted inside an inner scope bind
   to a deeper HeapRing (§5), so a long-lived outer-scope task cannot hold
   the FIFO head against inner-scope churn.

Every slot has a `fanout_total` counter: the number of outstanding lifetime
references (retaining consumers + any scope refs). A slot transitions to
`CONSUMED` (slot + heap slab freed) only when `fanout_released` meets the
threshold (`>= total + 1`; see §8 fanout-release threshold).

Without scope, a leaf task (no consumers, `fanout_total = 0`) would reach
COMPLETED but never transition further. Scope adds a deferred reference
that releases at `scope_end`:

```cpp
class Scope {
public:
    void    scope_begin();
    void    scope_end(const std::function<void(TaskSlot)> &release_ref);
    void    register_task(TaskSlot sid);    // called by Orchestrator.submit_*
    int32_t depth()         const;          // 1-based: 0 = no open scope
    int32_t current_depth() const;          // 0-based: L2-style; used for ring selection
private:
    std::vector<std::vector<TaskSlot>> stack_;
};
```

`Worker::run` always opens one outer scope; user orch fns may nest up to
`MAX_SCOPE_DEPTH = 64` additional scopes on top. Ring selection uses
the 0-based `current_depth()`:

| Where you are | `depth()` | `current_depth()` | Ring |
| ------------- | --------- | ----------------- | ---- |
| outer (Worker.run-only) scope | 1 | 0 | 0 |
| `with orch.scope():` | 2 | 1 | 1 |
| nested x 2 | 3 | 2 | 2 |
| nested x 3 | 4 | 3 | 3 |
| nested x 4 or deeper | >= 5 | >= 4 | 3 (clamp) |

Flow:

1. `scope_begin` pushes an empty frame onto `stack_`.
2. Each `submit_*` calls `scope.register_task(sid)`; the Orchestrator
   has already set `slot.fanout_total = scope_ref` (1 when `depth() > 0`)
   and stamped `slot.ring_idx = ring_idx_for_scope(current_depth())`
   before the call.
3. `scope_end` pops the frame; for each `sid`, invokes the release
   callback (`Orchestrator::release_ref`) which bumps `fanout_released`
   by 1 and transitions the slot to CONSUMED if the threshold is met.

### User-facing API

```python
def my_orch(orch, args, cfg):
    with orch.scope():                             # ring 1
        orch.submit_next_level(chip_a, a_args, cfg, worker=0)
        orch.submit_next_level(chip_b, b_args, cfg, worker=1)
    # Inner tasks are now eligible for reclamation on ring 1,
    # without waiting for any outer-scope task.
    orch.submit_next_level(chip_c, c_args, cfg, worker=0)  # ring 0
```

`with orch.scope():` is the recommended form. Raw `orch.scope_begin()` /
`orch.scope_end()` are exposed too, primarily for advanced use where the
context manager would be awkward (e.g., scopes that span a user-level
state machine).

### Why `scope_end` is non-blocking

`scope_end` walks the scope's tasks and bumps each one's `fanout_released`
counter, then returns. A task whose `fanout_released` now meets the
threshold transitions to CONSUMED inline; others stay COMPLETED or PENDING
until the scheduler and consumers finish their own releases. This mirrors
the chip runtime's `rt_scope_end`.

The internal run fence, not `scope_end`, provides synchronous completion.
`Worker.run` closes its outer scope, closes submission, and waits for that
run's active count to reach zero before returning. There is deliberately no
per-scope wait primitive.

---

## 7. TensorMap

The TensorMap maps `(RunId, TensorKey) → current_producer_slot`. It drives
automatic dependency inference without resolving a producer from another run.

```cpp
class TensorMap {
public:
    TaskSlot lookup(RunId run_id, TensorKey key) const;
    void insert(RunId run_id, TensorKey key, TaskSlot sid);
    void erase_task_outputs(RunId run_id, TaskSlot owner,
                            const std::vector<TensorKey> &keys);
private:
    std::mutex mu_;
    std::unordered_map<RunTensorKey, TaskSlot, RunTensorKeyHash> map_;
};
```

### Semantics

- **RAW (read-after-write)**: consumer's `INPUT` sees producer's `OUTPUT`
  entry → fanin edge recorded.
- **WAW (write-after-write)**: a new `OUTPUT` on the same address replaces
  the entry. The previous producer remains live (still has wire references
  from any prior consumers); new consumers depend only on the latest. The two
  writers carry no edge between them, so the earlier one can reach CONSUMED
  first — `erase_task_outputs` therefore drops a key only while it still maps
  to the consumed slot, leaving the later writer's entry for new consumers to
  find.
- **WAR (write-after-read)** is not tracked directly. Read tasks don't
  register in TensorMap; write tasks only look up current producer. If a
  consumer reads `X` (recording fanin on producer P1) and then a later task
  writes `X` (new producer P2 in TensorMap), there's no P1 → P2 edge. This is
  correct: the reader only needs P1 to have completed, the new writer only
  needs its own prior producer. Simultaneous read and write races are a user
  bug, not a scheduler concern. When a workload genuinely needs the reader
  ordered ahead of the overwrite, express it explicitly — see
  [WAR anti-dependencies](war-anti-dependency.md) (issue #1306) for the
  `add_dep` vs `INOUT` trade-off.

### Thread safety

TensorMap is written by the Orch thread in `submit_*` and modified by the
Scheduler thread when a slot becomes CONSUMED. A mutex serializes lookup,
insert, erase, and size operations across those threads.

---

## 8. Task State Machine

Each `TaskSlotState.state` progresses through:

```text
FREE ─► BUILDING ─► PENDING ──► READY ──► RUNNING ──► COMPLETED ──► CONSUMED ──► FREE
             │                │             │
             └────────────────┴─────────────┴──► FAILED ─────► CONSUMED ──► FREE
```

- **FREE**: slot in the ring pool, not allocated
- **BUILDING**: allocated; submit owns it. Step 3 registers its outputs in the
  TensorMap and step 5 appends it to its producers' fanout lists, so it is
  observable from other threads well before its fanin and fanout counters are
  final. No other thread may advance it — see
  [the publication rule](#the-building-publication-rule) below
- **PENDING**: published; waiting on live fanin producers
- **READY**: all fanins satisfied; queued for Scheduler dispatch
- **RUNNING**: dispatched to one or more endpoints
- **COMPLETED**: worker(s) done; may still be referenced by fanout / scope
- **FAILED**: a worker/endpoint failed, or a failed producer poisoned this
  slot. Failed slots are never dispatched if they were not already running.
- **CONSUMED**: all references released; Scheduler calls `ring.release(sid)`
  and the slot returns to FREE

State transitions are driven by atomic operations:

- Orch: FREE → BUILDING at slot claim, BUILDING → PENDING/READY at publication
- Scheduler: PENDING → READY → RUNNING during dispatch
- Scheduler: RUNNING → COMPLETED, or BUILDING/PENDING/READY → FAILED during
  completion / dependency poisoning
- Scheduler/Orch cleanup: COMPLETED/FAILED → CONSUMED

### The BUILDING publication rule

Wiring a task to a producer has to happen under that producer's `fanout_mu` —
otherwise a producer completing concurrently either misses the new consumer or
releases one that has not counted it. That makes a half-built task reachable
from another thread, and BUILDING is what tells that thread the counters are
not final:

- A **producer that fails** claims the slot through `claim_task_failure` and
  **stops there**. It does not mark group members, poison consumers, or release
  references — the submitting thread does, once its wiring is done. Running the
  propagation from both sides releases every producer reference twice, which
  reclaims a producer's output while the device is still reading it.
- A **producer that completes** advances `fanin_released` against a
  `fanin_count` submit has not published, and zero is a count any release
  passes. Readiness is therefore one decision, not two readable facts:
  `try_mark_ready` compares the pair and changes the state under `fanout_mu`,
  and publication writes `fanin_count` together with the transition out of
  BUILDING under the same lock. A producer that arrives first is re-evaluated
  by the publication; one that arrives after reads a published count. Neither
  can act on half of the pair, and exactly one enqueues the task.
- **Dispatch** never claims a BUILDING slot: there are no final args or fanin
  count to dispatch on.

`cancel_unstarted_run` is the one caller that *does* propagate for a BUILDING
slot, because submission is closed before it runs — a slot still BUILDING there
is one whose submit threw part-way through wiring, and no submitting thread is
left to publish it. The same applies one step later: a claim won from BUILDING
sets `failure_propagation_pending`, and cancellation takes over any slot still
carrying it. Without that, a submit that threw *after* being claimed would
leave a FAILED slot cancellation skips, holding references the run's fence
waits on forever.

Cancellation also records debt immediately after it wins a PENDING or READY
claim. Scheduler-owned PENDING/READY claims expose no takeover debt: their
claimant propagates exclusively, so cancellation cannot race it over stale slot
IDs.

Debt settlement is not the start of cancellation propagation. Cancellation
first marks groups and snapshots the producer lists for **every** slot it owns;
all of that work is idempotent and may allocate. An exception therefore leaves
each claimed slot's debt set and any unvisited slot claimable by the next
attempt. Only after every snapshot succeeds does cancellation clear the debts,
then perform the non-repeatable self and producer reference releases without
any further allocation.

### What a submit that throws leaves behind

A submit publishes BUILDING before its slot is charged to the run, so a throw
part-way has to leave either an unowned Ring slot it releases directly or a
registered slot the run's cancellation can fully reclaim. These ownership
rules carry that:

- **Run-slot registration is the ownership boundary.** If registration throws,
  the slot is absent from the run and cancellation cannot discover it, so
  submit marks it CONSUMED and directly releases the combined slot/HeapRing
  reservation before rethrowing. Once registration succeeds, all remaining
  fallible construction stays BUILDING and cancellation owns rollback.

- **The scope reference is registered before it is charged.** `fanout_total`
  counts a release that `scope_end` will make; charging it before
  `scope_->register_task` succeeded would leave a slot owing a release nothing
  can make. Cancellation contributes only the terminal release, the threshold
  is never reached, and the slot — and with it the run's task count and its
  fence — never resolves.
- **The charge immediately follows successful registration** and precedes Step
  2 publishing the slot's outputs. A failure between registration and charge
  leaves an extra scope release, which is safe; the opposite order would leave
  an unreleasable charge. Once an output is published, retaining consumers
  increment the same `fanout_total` field.
- **The output cleanup journal precedes TensorMap publication.** A failure can
  never leave a mapping cancellation does not know to erase, and the map's
  owner check preserves a previous producer when the replacement insert never
  completed.

### The failure claim

Every path that fails a task — a completion poisoning its consumers, a run
cancelling its unstarted slots, and a submit that wires onto an already-failed
producer — moves it to FAILED through `claim_task_failure`, and the winner is
the only one that writes `failure_message`. A plain store could put a CONSUMED
slot back to FAILED and then consume it, releasing its dependency references a
second time.

The claim, the message and the fanout snapshot are all taken under `fanout_mu`,
so a task wiring itself onto a producer either registers before the claim and is
poisoned by it, or observes FAILED — with its reason — and poisons itself. The
same lock covers the ordinary terminal transition in `on_task_complete`: split
from the snapshot, a consumer can wire itself in afterwards and never be
released, or read COMPLETED, decline to count the fanin, and still be released,
reaching READY one producer early.

### Fanout-release threshold

Both paths that can trigger COMPLETED/FAILED → CONSUMED (the scheduler's
`try_consume` and the scope-end `release_ref`) use the same threshold:

```cpp
if (fanout_released >= fanout_total + 1 &&
    (state == COMPLETED || state == FAILED))
    on_consumed(slot);
```

The `+1` accounts for the slot's own self-release contribution, which normal
tasks emit from `on_task_complete` (`try_consume(slot)` self-call). Alloc
slots (§8b) bypass the scheduler and pre-bump `fanout_released` to `1` at
`alloc()` time to stand in for the self-release. Both paths use `on_consumed`,
which CASes `state` from `COMPLETED` or `FAILED` to `CONSUMED` to remain
idempotent when both fire concurrently at threshold.

---

## 8b. `alloc(shape, dtype, identity)` — runtime-owned intermediate buffers

`alloc` creates a synthetic task slot in `COMPLETED` state that owns a
1024-byte-aligned slab of the Worker's HeapRing. The slab is reclaimed
implicitly once the slot reaches `CONSUMED` and `last_alive` sweeps over it
— no per-slot `munmap` runs.

```cpp
uint64_t Orchestrator::alloc(
    const std::vector<uint32_t> &shape, DataType dtype, const CanonicalIdentity &identity
) {
    // 1. Atomic {slot, heap_ptr} from the merged Ring. Blocks on
    //    back-pressure; throws on timeout.
    uint64_t aligned = align_up(nbytes(shape, dtype), HEAP_ALIGN);
    AllocResult ar = allocator_.alloc(aligned);
    TaskSlotState &s   = slots_[ar.slot];
    s.reset();
    // 2. Publish cancellation ownership before registering active_tasks. If
    //    run-slot registration itself throws, release the unowned Ring slot.
    s.run_id = current_run_id;
    s.state = TaskState::BUILDING;
    current_run.register_slot(ar.slot);
    // 3. No fanin — alloc has no work to wait on.
    s.fanin_count = 0;
    // 4. Register the scope reference before charging it. Consumers that wire
    //    on this slot in infer_deps later increment fanout_total.
    int32_t scope_ref = (scope_.depth() > 0) ? 1 : 0;
    if (scope_ref > 0) scope_.register_task(ar.slot);
    s.fanout_total = scope_ref;
    // 5. Key on the identity's canonical hash, so a Tensor carrying the same
    //    identity resolves to this slot in infer_deps. Journal the key before
    //    TensorMap publication so cancellation can erase every entry that
    //    became visible before a later failure.
    uint64_t key = CanonicalIdentityHash{}(identity);
    s.output_keys.push_back(key);
    tensormap_.insert(current_run_id, key, ar.slot);
    // 6. Sim self-consume so the fanout-release threshold math aligns with
    //    normal slots (see §8 Fanout-release threshold).
    s.fanout_released = 1;
    // 7. Straight to COMPLETED — no dispatch needed.
    s.state = TaskState::COMPLETED;
    return reinterpret_cast<uint64_t>(ar.heap_ptr);
}
```

All fallible setup after run registration happens while the synthetic slot is
BUILDING. If `alloc()` throws, graph-failure cancellation can therefore claim
and consume it just like an interrupted task submit. The synthetic self-release
is published only after those fallible steps; a failed BUILDING slot gets that
terminal release from cancellation instead. Run-slot registration is the one
earlier boundary cancellation cannot cover, so failure there directly releases
the Ring claim before rethrowing.

`on_consumed` runs the usual `tensormap.erase_task_outputs` and then calls
`allocator_.release(sid)`. FIFO reclamation inside the allocator returns the
slab to the heap's free region as `last_alive` advances; callers see no
per-slab free syscall.

### Consumer interaction

`infer_deps` treats `COMPLETED` producers specially. A retained tensor or
`add_dep` edge still holds the producer until the consumer finishes, but
contributes no live fanin. An ordering-only `add_dep_wait` is already satisfied
and adds no lifetime reference. A producer that is already `FAILED` poisons
downstream consumers instead of dispatching them.

```cpp
if (ps_state == TaskState::CONSUMED) continue;  // already gone
if (!dep.retain && ps_state == TaskState::COMPLETED) continue;
ps.fanout_consumers.push_back(slot);
if (dep.retain) {
    ps.fanout_total++;
    s.fanin_producers.push_back(prod);
}
if (ps_state != TaskState::COMPLETED) live_fanins++;   // wait only if not yet done
```

### Tag semantics for write-after-write

`infer_deps` mirrors L2 (`orchestrator.cpp` Step B): only `INPUT`
and `INOUT` do a tensormap lookup. `OUTPUT` and `OUTPUT_EXISTING`
are pure inserts — the latter is the way users signal "skip the
lookup even though I'm writing a pre-existing buffer".

| Tag | TensorMap lookup | TensorMap insert | Dep wired on prior owner |
| --- | ---------------- | ---------------- | ------------------------ |
| `INPUT` | ✓ | — | RaW |
| `INOUT` | ✓ | ✓ | RaW + WaW |
| `OUTPUT` | — | ✓ | **none** — pure overwrite |
| `OUTPUT_EXISTING` | — | ✓ | **none** — pure overwrite, skips lookup |
| `NO_DEP` | — | — | — |

A task that writes into a buffer handed out by `orch.alloc()` and
needs the alloc-slot to stay live while it writes must tag the
tensor `INOUT`. `INOUT` is the only tag that pulls the creator in
as a fanin producer, pinning the alloc-slot against reclamation.
Tagging the same buffer `OUTPUT` / `OUTPUT_EXISTING` is a pure
overwrite and leaves no lifetime link: if the caller needs the
buffer to outlive the creator they must maintain that lifetime
themselves.

### `OUTPUT` auto-allocation

If an `OUTPUT`-tagged tensor arrives at `submit_*` with `data == 0`, the
Orchestrator reserves a slab from the HeapRing as part of the same
`Ring::alloc` call that claims the slot. All OUTPUT slabs for a
single submit share one `alloc(total_bytes)` call — the returned base
pointer is carved into per-tensor slabs, each 1024-byte aligned.
OUTPUT tensors whose `data` is already set are left alone (legacy
"user-provided buffer" path, and the entry point for
`orch.alloc()`-then-submit). `OUTPUT_EXISTING` is never auto-allocated.

### `heap_ring_size` and back-pressure

The HeapRing size is a `Worker` ctor parameter, surfaced on the Python
`Worker` as `heap_ring_size=` (default 1 GiB). The heap is `mmap`'d in the
C++ ctor — before Python forks the chip / sub child processes — so the
children inherit the same `MAP_SHARED | MAP_ANONYMOUS` region at the same
virtual address.

When the heap or the slot window is full, `allocator.alloc()` spin-waits on
the shared cv. If the `timeout_ms` elapses with no progress, it throws
`std::runtime_error` (typical wrappers: `"HeapRing exhausted, increase
heap_ring_size on Worker"` or `"task window full"`). That bubbles out of
`Worker.run` as a Python exception so users can recover or grow the ring
instead of stalling forever. Default timeout: 10 s.

## 8c. Fork hygiene

`Worker`'s ctor runs a one-shot `fork_hygiene_once()` step before it
`mmap`s the heap. Two pieces:

1. **Thread-pool env defaults** — `setenv` with `overwrite=0`:
   - `OMP_NUM_THREADS=1`
   - `OPENBLAS_NUM_THREADS=1`
   - `MKL_NUM_THREADS=1`
   - `BLIS_NUM_THREADS=1`
   - `KMP_DUPLICATE_LIB_OK=TRUE` (macOS only, tolerates duplicate libomp
     loads across Python / PyTorch / NumPy)

   These keep transitively-linked thread pools from spinning up worker
   threads we would then inherit across `fork()`. User-supplied values win.

2. **`pthread_atfork`** handler registered once per process. The handler is
   currently a landing pad; the Allocator's mutex is the only Worker-owned
   lock that matters today, and it is not held across any fork point. The
   handler documents the acquisition order we'll use as more locks are added
   in subsequent PRs (callable registry → worker manager → worker thread →
   scheduler → allocator → tensormap, coarse-to-fine).

---

## 9. Invariants

1. **Orch is single-threaded**: only one thread builds a run and calls
   `submit_*` at a time. TensorMap still uses a mutex because Scheduler-driven
   consumption erases entries concurrently.
2. **Tags are consumed at submit**: `task_args.tag(i)` is read only inside
   `submit_*`. Phases after submit (slot storage, dispatch, execution) do not
   see tags.
3. **Slot is parent-heap**: all `TaskSlotState` state is in the parent
   process's heap. Forked child workers never read slot state; they
   receive task data via the mailbox (see
   [worker-manager.md](worker-manager.md) §3).
4. **Ring.alloc is the only blocking point in Orch**: `submit_*` never
   blocks except on ring pressure.
5. **Scope.register_task is idempotent per slot per scope level**: each
   submitted slot gets exactly one scope ref at its current scope depth.

---

## 10. Related

- [hierarchical-level-runtime.md](hierarchical-level-runtime.md) — how
  Orchestrator fits alongside Scheduler and Worker
- [scheduler.md](scheduler.md) — READY dispatch and completion-time dependency
  release
- [task-flow.md](task-flow.md) — the data (Callable / TaskArgs / CallConfig)
  being moved by `submit_*`
- [comm-domain.md](comm-domain.md) — `orch.allocate_domain` dynamic
  communication-domain allocation, lifetime model, and backends
