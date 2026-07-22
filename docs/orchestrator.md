# Orchestrator — DAG Submission Internals

Callable identity update: the Python facade validates `CallableHandle` objects
and passes callable digest/kind/namespace metadata into the C++ Orchestrator.
Older `callable_id`/`cid` examples below are target-local or historical
internals, not public submit arguments. See
[callable-identity-registration.md](callable-identity-registration.md).

The Orchestrator is the **DAG builder**. It runs single-threaded on the user's
thread (inside `Worker::run` between `scope_begin` and `drain`) and owns the
three data structures that turn a sequence of `submit_*` calls into a scheduled
DAG: `Ring`, `TensorMap`, and `Scope`.

For the high-level role of the Orchestrator among the three engine components,
see [hierarchical_level_runtime.md](hierarchical_level_runtime.md). For what
flows through `submit`, see [task-flow.md](task-flow.md).

---

## 1. Python Facade and C++ Internal API

The Python user's orch fn receives a `simpler.orchestrator.Orchestrator`
facade. Its `submit_*` methods enqueue DAG nodes and return `None`; task slots
remain internal to the worker.

The C++ Orchestrator still returns `SubmitResult` for internal scheduling and
C++ tests, but nanobind intentionally drops that return value instead of
exposing it to Python:

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
    Tensor alloc(const std::vector<uint32_t> &shape, DataType dtype);

    // --- Internal lifecycle (invoked by Worker::run only, bound as _scope_begin
    //     / _scope_end / _drain in the Python facade) ---
    void scope_begin();
    void scope_end();
    void drain();

private:
    // ... components: Ring, TensorMap, Scope, slot pool, active_tasks_ counter
};

struct SubmitResult { TaskSlot task_slot; };  // internal only; not bound to Python
```

**Status**: `submit_sub` takes only `(CallableIdentity, args)` — no
`config`, since SUB has no per-call config.

`scope_begin` / `scope_end` / `drain` are invoked from Python `Worker.run` via
`_scope_begin` / `_scope_end` / `_drain` bindings. They are not part of the
user-facing orch-fn API.

Remote L3 submit adds two hidden pieces of metadata: final eligible worker-id
sets and optional `RemoteTaskArgsSidecar` entries aligned by tensor index.
Python `RemoteCallable` handles supply callable eligibility, and
`TaskArgs.add_tensor(RemoteTensorRef(...), tag)` supplies tensor sidecars. The
Orchestrator validates exact placement, worker existence, local-vs-remote
compatibility, remote handle access rights for the tensor tag, bare host
pointers, and remote null OUTPUT tensors before committing the slot.
For NEXT_LEVEL tasks, `worker`/`workers` are required stable worker ids rather
than C++ worker-thread vector indices. SUB submit APIs remain unconstrained.

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

    // 3. Walk task_args tags, derive dependencies
    //    (dedup producers: same producer may appear on multiple input tensors)
    std::vector<TaskSlot> producers;
    std::unordered_set<TaskSlot> producers_seen;
    for (int i = 0; i < s.task_args.tensor_count(); i++) {
        TensorArgType tag = s.task_args.tag(i);
        TensorKey key     = key_for_tensor_or_remote_sidecar(i);

        if (tag == INPUT || tag == INOUT) {
            if (TaskSlot prod = tensormap_.lookup(key); prod != INVALID)
                if (producers_seen.insert(prod).second)
                    producers.push_back(prod);
        }
        if (tag == OUTPUT || tag == INOUT || tag == OUTPUT_EXISTING) {
            tensormap_.insert(key, sid);
        }
        // NO_DEP: skip both
    }

    // 4. Record fanin on self
    s.fanin_count    = static_cast<int32_t>(producers.size());
    s.fanin_released = 0;

    // 5. Register with scope (holds slot open until scope_end releases ref)
    scope_.register_task(sid);          // increments s.fanout_total by 1

    // 6. Push fanout edges onto scheduler's wiring queue
    //    (Scheduler wires producer→consumer asynchronously; avoids blocking
    //    the Orch thread on fanout_mu)
    scheduler_.enqueue_wiring(sid, std::move(producers));

    // 7. Return handle
    return {sid};
}
```

### Step details

**Step 1 — `ring_.alloc()`**: See [§5 Ring](#5-ring-slot--per-scope-heap-allocator). Blocks the Orch thread
if all slots are in-flight; this is the system's back-pressure mechanism.

**Step 2 — store task data**: `TaskArgs` is moved (not copied). `config` is a
small POD copied by value. `callable` is a `uint64_t` opaque handle (see
[task-flow.md](task-flow.md) §2).

**Step 3 — tag walk**: The only place tags are consumed. After this step tags
are never inspected again; they are not carried into the slot's stored
`task_args` value during dispatch (see [task-flow.md](task-flow.md) §3).
Local tensors key TensorMap by `(LOCAL_HOST, ptr)` or
`(LOCAL_CHILD, worker, ptr)`. Remote tensors with sidecars key by
`(address_kind, owner_worker_id, buffer_id, generation, offset)`.

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

**Step 4 — fanin count**: The number of live producers. Decremented by
`fanin_released++` each time a producer completes; when `fanin_released ==
fanin_count`, the slot is ready. A ready NEXT_LEVEL single task is routed to
the FIFO for its required stable worker id. The same routing function is used
for immediately-ready submissions and tasks released by Scheduler dependency
processing. NEXT_LEVEL groups and SUB tasks remain on their shared queues in
this transition.

**Step 5 — scope ref**: Each slot starts with one "scope reference" in its
fanout_total. Without this, a task with no downstream consumer would never be
reclaimable. See [§6 Scope](#6-scope).

**Step 6 — wiring queue**: Fanout edges (producer knows its consumers) are
wired **asynchronously** by the Scheduler thread. This decouples submit from
`fanout_mu` contention. See [scheduler.md](scheduler.md) §2 for the wiring
phase.

---

## 3. `submit_next_level_group` — N workers, 1 DAG node

A group task is a single DAG node that executes in parallel on N workers.
Each worker gets its own `TaskArgs`; the node only reaches COMPLETED when all
N finish.

```cpp
SubmitResult Orchestrator::submit_next_level_group(const CallableIdentity &callable,
                                                    std::vector<TaskArgs> args_list,
                                                    const CallConfig &config) {
    TaskSlot sid = ring_.alloc();
    TaskSlotState &s = slots_[sid];
    s.reset();
    s.worker_type     = WorkerType::NEXT_LEVEL;
    s.callable        = callable;
    s.config          = config;
    s.group_size      = args_list.size();
    s.sub_complete_count = 0;
    s.task_args_list  = std::move(args_list);

    // Tag walk unions all entries in args_list (any input in any member → fanin)
    // Dedup both producers and outputs across all args_list entries.
    std::vector<TaskSlot> producers;
    std::unordered_set<TaskSlot> producers_seen;
    std::unordered_set<uint64_t> outputs_seen;
    for (auto &a : s.task_args_list) {
        for (int i = 0; i < a.tensor_count(); i++) {
            TensorArgType tag = a.tag(i);
            uint64_t ptr      = a.tensor(i).data;
            if (tag == INPUT || tag == INOUT)
                if (auto prod = tensormap_.lookup(ptr); prod != INVALID)
                    if (producers_seen.insert(prod).second)
                        producers.push_back(prod);
            if (tag == OUTPUT || tag == INOUT || tag == OUTPUT_EXISTING)
                if (outputs_seen.insert(ptr).second)
                    tensormap_.insert(ptr, sid);
        }
    }

    s.fanin_count    = static_cast<int32_t>(producers.size());
    s.fanin_released = 0;
    scope_.register_task(sid);
    scheduler_.enqueue_wiring(sid, std::move(producers));
    return {sid};
}
```

At dispatch time the Scheduler reserves `group_size` idle WorkerThreads, and
each WorkerThread runs `worker->run` with its own `task_args_list[i]`.
Completion is gated on `sub_complete_count.fetch_add(1) + 1 == group_size`.

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

## 5. Ring (slot + per-scope heap allocator)

`Ring` owns three correlated per-task resources:

1. A **monotonic task id** — allocated on every `alloc()`, shared across
   all rings. There is no fixed window and no modulo wrap at L3: slot
   state lives in parent-process heap (never crossed into child workers),
   so the ring-index addressing scheme L2 needs for shmem descriptors
   buys us nothing here. A monotonic `int32_t` gives ~2 billion ids per
   `reset_to_empty()` interval, reset to 0 at the end of every
   `Worker.run()`.
2. **`MAX_RING_DEPTH = 4` independent shared-memory heap slabs**
   (Strict-1; matches L2's `PTO2_MAX_RING_DEPTH`). Each slab has its own
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
   stays valid until `reset_to_empty()` drops the whole deque.

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
(matches L2's `PTO2_PACKED_OUTPUT_ALIGN`, Strict-3).

**FIFO reclamation per ring**: each `alloc()` appends the slot's
`heap_end_offset` onto the selected ring's `slot_heap_end[]` vector, and
pushes a `released=0` byte. `release(slot)` looks up the slot's ring via
`slot.ring_idx` and advances **that ring's** `last_alive` as long as the
next-oldest in-ring slot is released, walking the ring's `heap_tail`
forward. Rings never touch each other — inner-scope tasks reclaim
without waiting for an outer-scope task to finish.

**End-of-run reset**: `Orchestrator::drain()` waits for
`active_tasks_` to hit 0, then calls `ring.reset_to_empty()` which
drops the whole slot-state deque *and* rewinds every ring's cursors /
`released[]` / `slot_heap_end[]` back to 0. Memory per `Worker.run()`
is bounded by that run's peak alive task count; nothing accumulates
across runs.

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

Every slot has a `fanout_total` counter: the number of outstanding
references (downstream consumers + any scope refs). A slot transitions to
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
L2's `pto2_scope_end`.

Users who need a synchronous wait for *all* in-flight tasks must call
`drain()` (or let `Worker::run` finish — its outer `scope_end` is followed
by `drain()` before the call returns). There is deliberately no
per-scope drain primitive: the extra machinery (per-scope active counter
and cv) would only pay for itself in patterns we do not have yet.

---

## 7. TensorMap

The TensorMap maps `tensor_base_ptr → current_producer_slot`. It drives
automatic dependency inference.

```cpp
class TensorMap {
public:
    TaskSlot lookup(uint64_t base_ptr) const;         // returns INVALID if absent
    void     insert(uint64_t base_ptr, TaskSlot sid); // overwrites; previous
                                                      // producer remains wire-referenced
    void     erase(uint64_t base_ptr);                // called when producer
                                                      // reaches CONSUMED
private:
    std::unordered_map<uint64_t, TaskSlot> map_;
};
```

### Semantics

- **RAW (read-after-write)**: consumer's `INPUT` sees producer's `OUTPUT`
  entry → fanin edge recorded.
- **WAW (write-after-write)**: a new `OUTPUT` on the same address replaces
  the entry. The previous producer remains live (still has wire references
  from any prior consumers); new consumers depend only on the latest.
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

TensorMap is written only by the Orch thread (in `submit_*`) and modified by
the Scheduler thread via `erase` (on CONSUMED). Since `submit_*` and `erase`
for different entries are non-overlapping in practice, a single mutex guards
the map in the current implementation. If contention becomes a concern, a
concurrent hash map can replace it.

---

## 8. Task State Machine

Each `TaskSlotState.state` progresses through:

```text
FREE ──► PENDING ──► READY ──► RUNNING ──► COMPLETED ──► CONSUMED ──► FREE
                             │               │
                             └──────────────► FAILED ─────► CONSUMED ──► FREE
```

- **FREE**: slot in the ring pool, not allocated
- **PENDING**: allocated; waiting on live fanin producers
- **READY**: all fanins satisfied; queued for Scheduler dispatch
- **RUNNING**: dispatched to one or more endpoints
- **COMPLETED**: worker(s) done; may still be referenced by fanout / scope
- **FAILED**: a worker/endpoint failed, or a failed producer poisoned this
  slot. Failed slots are never dispatched if they were not already running.
- **CONSUMED**: all references released; Scheduler calls `ring.release(sid)`
  and the slot returns to FREE

State transitions are driven by atomic operations:

- Orch: FREE → PENDING at submit time
- Scheduler: PENDING → READY → RUNNING during dispatch
- Scheduler: RUNNING → COMPLETED or RUNNING/PENDING/READY → FAILED during
  completion / dependency poisoning
- Scheduler/Orch cleanup: COMPLETED/FAILED → CONSUMED

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

## 8b. `alloc(shape, dtype)` — runtime-owned intermediate buffers

`alloc` creates a synthetic task slot in `COMPLETED` state that owns a
1024-byte-aligned slab of the Worker's HeapRing. The slab is reclaimed
implicitly once the slot reaches `CONSUMED` and `last_alive` sweeps over it
— no per-slot `munmap` runs.

```cpp
Tensor Orchestrator::alloc(const std::vector<uint32_t> &shape, DataType dtype) {
    // 1. Atomic {slot, heap_ptr} from the merged Ring. Blocks on
    //    back-pressure; throws on timeout.
    uint64_t aligned = align_up(nbytes(shape, dtype), HEAP_ALIGN);
    AllocResult ar = allocator_.alloc(aligned);
    TaskSlotState &s   = slots_[ar.slot];
    s.reset();
    // 2. Register as this slot's output so downstream tensors with the same
    //    data pointer look up this slot as producer.
    uint64_t key = reinterpret_cast<uint64_t>(ar.heap_ptr);
    tensormap_.insert(key, ar.slot);
    s.output_keys.push_back(key);
    // 3. No fanin — alloc has no work to wait on.
    s.fanin_count = 0;
    // 4. Initial fanout = scope_ref. Consumers that wire on this slot in
    //    infer_deps bump fanout_total; this slot's CONSUMED transition waits
    //    for all of them + scope_end.
    s.fanout_total = (scope_.depth() > 0) ? 1 : 0;
    if (s.fanout_total > 0) scope_.register_task(ar.slot);
    // 5. Sim self-consume so the fanout-release threshold math aligns with
    //    normal slots (see §8 Fanout-release threshold).
    s.fanout_released = 1;
    // 6. Straight to COMPLETED — no dispatch needed.
    s.state = TaskState::COMPLETED;
    active_tasks_++;
    return Tensor{key, shape, dtype};
}
```

`on_consumed` runs the usual `tensormap.erase_task_outputs` and then calls
`allocator_.release(sid)`. FIFO reclamation inside the allocator returns the
slab to the heap's free region as `last_alive` advances; callers see no
per-slab free syscall.

### Consumer interaction

`infer_deps` treats `COMPLETED` producers specially: it still wires the
fanout edge (so the producer waits for the consumer before being consumed and
freeing its buffer) but does not bump `live_fanins` (the consumer is
immediately ready because the producer is already done). A producer that is
already `FAILED` is not a successful fanin; downstream consumers are poisoned
by the Scheduler rather than dispatched.

```cpp
if (ps_state == TaskState::CONSUMED) continue;  // already gone
ps.fanout_consumers.push_back(slot);
ps.fanout_total++;
s.fanin_producers.push_back(prod);
if (ps_state != TaskState::COMPLETED) live_fanins++;   // wait only if not yet done
```

### Tag semantics for write-after-write

`infer_deps` mirrors L2 (`pto_orchestrator.cpp` Step B): only `INPUT`
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

1. **Orch is single-threaded**: only one thread ever calls `submit_*` or holds
   the `Orchestrator`. No locking is needed on TensorMap, Scope, or Ring-head
   for self-writes.
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

- [hierarchical_level_runtime.md](hierarchical_level_runtime.md) — how
  Orchestrator fits alongside Scheduler and Worker
- [scheduler.md](scheduler.md) — what happens to slots after they're pushed
  onto the wiring queue
- [task-flow.md](task-flow.md) — the data (Callable / TaskArgs / CallConfig)
  being moved by `submit_*`
- [comm-domain.md](comm-domain.md) — `orch.allocate_domain` dynamic
  communication-domain allocation, lifetime model, and backends
