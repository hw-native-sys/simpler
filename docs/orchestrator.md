# Orchestrator — DAG Submission Internals

> **Status**: describes the **target** design. Current code matches the
> user-facing submit API and `alloc` surface; inline "Status:" notes flag
> the few remaining divergences. See [roadmap.md](roadmap.md) for the
> full landed-vs-planned breakdown.

The Orchestrator is the **DAG builder**. It runs single-threaded on the user's
thread (inside `Worker::run` between `scope_begin` and `drain`) and owns the
three data structures that turn a sequence of `submit_*` calls into a scheduled
DAG: `Ring`, `TensorMap`, and `Scope`.

For the high-level role of the Orchestrator among the three engine components,
see [distributed_level_runtime.md](distributed_level_runtime.md). For what
flows through `submit`, see [task-flow.md](task-flow.md).

---

## 1. Public API

The user's orch fn receives an `Orchestrator*` as its first argument:

```cpp
class Orchestrator {
public:
    // --- User-facing submit API (tags inside TaskArgs drive deps) ---
    SubmitResult submit_next_level(uint64_t callable,
                                    const TaskArgs &args,
                                    const ChipCallConfig &config);
    SubmitResult submit_next_level_group(uint64_t callable,
                                          const std::vector<TaskArgs> &args_list,
                                          const ChipCallConfig &config);
    SubmitResult submit_sub(int32_t callable_id, const TaskArgs &args);
    SubmitResult submit_sub_group(int32_t callable_id,
                                   const std::vector<TaskArgs> &args_list);

    // --- Intermediate-buffer allocation (runtime-owned lifetime) ---
    ContinuousTensor alloc(const std::vector<uint32_t> &shape, DataType dtype);

    // --- Internal lifecycle (invoked by Worker::run only, bound as _scope_begin
    //     / _scope_end / _drain in the Python facade) ---
    void scope_begin();
    void scope_end();
    void drain();

private:
    // ... components: Ring, TensorMap, Scope, slot pool, active_tasks_ counter
};

struct SubmitResult { TaskSlot task_slot; };  // field is `task_slot` in current code
```

**Status**: `submit_sub` takes only `(callable_id, args)` — no `config`, SUB
has no per-call config. Target design (plan §"Why L2 has no submit") allows
callable IDs that may later unify with ChipCallable pointers; see PR-E.

`scope_begin` / `scope_end` / `drain` are invoked from Python `Worker.run` via
`_scope_begin` / `_scope_end` / `_drain` bindings. They are not part of the
user-facing orch-fn API.

---

## 2. `submit_next_level` — the 7-step flow

This is the entry point for every task in the DAG. All submit variants share
the same skeleton; `submit_next_level_group` and `submit_sub` differ only in
how the slot is set up.

```cpp
SubmitResult Orchestrator::submit_next_level(Callable cb,
                                              TaskArgs args,
                                              const CallConfig &config) {
    // 1. Alloc slot (blocks on back-pressure if ring full)
    TaskSlot sid = ring_.alloc();
    TaskSlotState &s = slots_[sid];
    s.reset();

    // 2. Move task data into slot (parent heap, no encoding)
    s.worker_type = WorkerType::NEXT_LEVEL;
    s.callable    = cb;
    s.task_args   = std::move(args);
    s.config      = config;

    // 3. Walk task_args tags, derive dependencies
    //    (dedup producers: same producer may appear on multiple input tensors)
    std::vector<TaskSlot> producers;
    std::unordered_set<TaskSlot> producers_seen;
    for (int i = 0; i < s.task_args.tensor_count(); i++) {
        TensorArgType tag = s.task_args.tag(i);
        uint64_t ptr      = s.task_args.tensor(i).data;

        if (tag == INPUT || tag == INOUT) {
            if (TaskSlot prod = tensormap_.lookup(ptr); prod != INVALID)
                if (producers_seen.insert(prod).second)
                    producers.push_back(prod);
        }
        if (tag == OUTPUT || tag == INOUT || tag == OUTPUT_EXISTING) {
            tensormap_.insert(ptr, sid);
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

**Step 1 — `ring_.alloc()`**: See [§5 Ring](#5-ring). Blocks the Orch thread
if all slots are in-flight; this is the system's back-pressure mechanism.

**Step 2 — store task data**: `TaskArgs` is moved (not copied). `config` is a
small POD copied by value. `callable` is a `uint64_t` opaque handle (see
[task-flow.md](task-flow.md) §2).

**Step 3 — tag walk**: The only place tags are consumed. After this step tags
are never inspected again; they are not carried into the slot's stored
`task_args` value during dispatch (see [task-flow.md](task-flow.md) §3).

| Tag | `tensormap.lookup` | `tensormap.insert` |
| --- | ------------------ | ------------------ |
| `INPUT` | ✓ | — |
| `OUTPUT` | — | ✓ |
| `INOUT` | ✓ | ✓ |
| `OUTPUT_EXISTING` | — | ✓ |
| `NO_DEP` | — | — |

`OUTPUT_EXISTING` differs from `OUTPUT` in runtime semantics (user-provided
buffer vs. runtime-allocated) but dependency tracking is identical: both
register this task as the new producer of `tensor.data`.

**Step 4 — fanin count**: The number of live producers. Decremented by
`fanin_released++` each time a producer completes; when `fanin_released ==
fanin_count`, the slot is ready.

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
SubmitResult Orchestrator::submit_next_level_group(Callable cb,
                                                    std::vector<TaskArgs> args_list,
                                                    const CallConfig &config) {
    TaskSlot sid = ring_.alloc();
    TaskSlotState &s = slots_[sid];
    s.reset();
    s.worker_type     = WorkerType::NEXT_LEVEL;
    s.callable        = cb;
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

Sub tasks have no C++ callable — they look up a Python function by id:

```cpp
SubmitResult Orchestrator::submit_sub(Callable cb, TaskArgs args, const CallConfig &config) {
    TaskSlot sid = ring_.alloc();
    TaskSlotState &s = slots_[sid];
    s.reset();
    s.worker_type = WorkerType::SUB;
    s.callable    = cb;                 // interpreted as callable_id
    s.task_args   = std::move(args);
    s.config      = config;

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

## 5. Ring

The `Ring` is a fixed-size slot pool with back-pressure.

```cpp
class Ring {
public:
    explicit Ring(int32_t window_size);   // typical: 128

    TaskSlot alloc();                      // blocks if full
    void     release(TaskSlot sid);        // called by Scheduler on CONSUMED

private:
    TaskSlotState *slots_;
    int32_t size_;
    std::atomic<uint32_t> head_;           // orch-only, next to alloc
    std::atomic<uint32_t> tail_;           // scheduler-only, next to release
    std::mutex mu_;
    std::condition_variable cv_;
};
```

**Back-pressure rationale**: if the Orch thread submits tasks faster than the
Scheduler + Workers can drain them, slots pile up. A fixed window forces the
Orch thread to pause, preventing unbounded memory growth and keeping DAG
depth reasonable.

**Ownership by role**:

| Field | Writer | Reader |
| ----- | ------ | ------ |
| `head_` | Orch (`alloc`) | Orch only |
| `tail_` | Scheduler (`release`) | Scheduler only |
| `slots_[i]` | Orch at submit, Scheduler on completion | both per-phase |

Orch and Scheduler coordinate via per-slot atomics (`state`, `fanin_released`)
and per-slot mutexes (`fanout_mu`), not via ring-level atomics beyond the
head/tail positions.

---

## 6. Scope

Scope solves: **"how do we release a task's ring slot if it has no downstream
consumer?"**

Every slot has a `fanout_total` counter: the number of outstanding references
(downstream consumers + any scope refs). A slot transitions to `CONSUMED`
(ring slot freed) only when `fanout_total == fanout_released`.

Without scope, a leaf task (no consumers, `fanout_total = 0`) would reach
COMPLETED but never transition further — but then all its outputs have been
observed at the earliest moment, so it's actually fine in this degenerate
case. The problem appears when user code does this:

```python
def my_orch(orch, args, cfg):
    r = orch.submit_next_level(...)    # produces tensor X
    # no one consumes X within this DAG
    # without scope: slot stays, ring fills up eventually
```

Scope adds a deferred reference that releases at `scope_end`:

```cpp
class Scope {
public:
    void scope_begin();
    void scope_end(const std::function<void(TaskSlot)> &release_ref);
    void register_task(TaskSlot sid);       // called by Orchestrator.submit_*
private:
    std::vector<std::vector<TaskSlot>> depth_;   // stack of scope levels
};
```

Flow:

1. `scope_begin` pushes an empty vector onto the depth stack
2. Each `submit_*` calls `scope.register_task(sid)`, appending to the top
   vector and bumping `slots_[sid].fanout_total` by 1
3. `scope_end` pops the top vector; for each `sid`, releases the scope ref
   (`release_ref(sid)` decrements `fanout_total` bookkeeping and may
   transition the slot to CONSUMED)

Nested scopes are supported (the stack structure). For now only `Worker::run`
opens a single top-level scope; nested scopes would be a future extension for
explicit user scoping.

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
  bug, not a scheduler concern.

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
 ↑         │           │          │            │             │
 │       submit       fanin=0   Scheduler    worker        all refs
 │       has fanin    (submit   dispatches   done          released
 │                    or fanout  to WT                     (scope +
 │                    release)                              fanout)
 │                                                          │
 └──────────────────── ring.release(sid) ◄─────────────────┘
```

- **FREE**: slot in the ring pool, not allocated
- **PENDING**: allocated, `fanin_count > 0`, waiting on producers
- **READY**: pushed to ready_queue (will be dispatched)
- **RUNNING**: Scheduler has dispatched to a WorkerThread; for group tasks,
  `sub_complete_count < group_size`
- **COMPLETED**: worker(s) done; may still be referenced by fanout / scope
- **CONSUMED**: all references released; Scheduler calls `ring.release(sid)`
  and the slot returns to FREE

State transitions are driven by atomic CAS operations:

- Orch: FREE → PENDING/READY at submit time
- Scheduler: READY → RUNNING → COMPLETED → CONSUMED during dispatch/completion

### Fanout-release threshold

Both paths that can trigger COMPLETED → CONSUMED (the scheduler's
`try_consume` and the scope-end `release_ref`) use the same threshold:

```cpp
if (fanout_released >= fanout_total + 1 && state == COMPLETED) on_consumed(slot);
```

The `+1` accounts for the slot's own self-release contribution, which normal
tasks emit from `on_task_complete` (`try_consume(slot)` self-call). Alloc
slots (§8b) bypass the scheduler and pre-bump `fanout_released` to `1` at
`alloc()` time to stand in for the self-release. Both paths use `on_consumed`,
which uses a CAS on `state` from `COMPLETED` to `CONSUMED` to remain idempotent
when both fire concurrently at threshold.

---

## 8b. `alloc(shape, dtype)` — runtime-owned intermediate buffers

Mirrors L2's "task slot owns its output buffer" model: `alloc` creates a
synthetic task slot in `COMPLETED` state that owns an mmap'd buffer. The
buffer is freed when the slot reaches `CONSUMED` — i.e. after all downstream
consumers have completed and the scope ref has been released.

```cpp
ContinuousTensor Orchestrator::alloc(const std::vector<uint32_t> &shape, DataType dtype) {
    // 1. mmap(MAP_SHARED|MAP_ANONYMOUS) a page-aligned region — visible to
    //    forked child workers at the same virtual address.
    void *buf = mmap(...);
    // 2. Claim a task slot.
    TaskSlot sid = ring_.alloc();
    TaskSlotState &s = slots_[sid];
    // 3. Record buffer for on_consumed munmap.
    s.alloc_bufs.push_back(buf);
    s.alloc_sizes.push_back(mmap_bytes);
    // 4. Register as this slot's output so downstream `INPUT`-tagged tensors
    //    with the same data ptr look up this slot as producer.
    tensormap_.insert(reinterpret_cast<uint64_t>(buf), sid);
    s.output_keys.push_back(reinterpret_cast<uint64_t>(buf));
    // 5. No fanin — alloc has no work to wait on.
    s.fanin_count = 0;
    // 6. Initial fanout = scope_ref. Consumers that wire on this slot in
    //    infer_deps bump fanout_total; this slot's CONSUMED transition waits
    //    for all of them + scope_end.
    s.fanout_total = (scope_.depth() > 0) ? 1 : 0;
    if (s.fanout_total > 0) scope_.register_task(sid);
    // 7. Sim self-consume so the fanout-release threshold math aligns with
    //    normal slots (see §8 Fanout-release threshold).
    s.fanout_released = 1;
    // 8. Straight to COMPLETED — no dispatch needed.
    s.state = TaskState::COMPLETED;
    active_tasks_++;
    return ContinuousTensor{buf, shape, dtype};
}
```

On `on_consumed`, in addition to the usual `tensormap.erase_task_outputs` and
`ring.release(sid)`, the slot's `alloc_bufs` are `munmap`'d.

### Consumer interaction

`infer_deps` treats `COMPLETED` producers specially: it still wires the
fanout edge (so the producer waits for the consumer before being consumed and
freeing its buffer) but does not bump `live_fanins` (the consumer is
immediately ready because the producer is already done).

```cpp
if (ps_state == TaskState::CONSUMED) continue;  // already gone
ps.fanout_consumers.push_back(slot);
ps.fanout_total++;
s.fanin_producers.push_back(prod);
if (ps_state != TaskState::COMPLETED) live_fanins++;   // wait only if not yet done
```

### Status — placeholder vs target (PR-H)

The current implementation uses **per-alloc `mmap`** (one syscall per
`alloc()` invocation). This is a placeholder. The target design (PR-H,
"HeapRing") pre-allocates a single MAP_SHARED region at `Worker::init()`
before any fork, bump-allocates from it, and reclaims via FIFO
`last_alive` tracking — mirroring L2's `PTO2TaskAllocator`. Under the
target design, `OUTPUT`-tagged tensors will be auto-allocated by the
Orchestrator (no explicit `alloc` call), and `OUTPUT_EXISTING` will
preserve the current "user-provided buffer" path.

---

## 9. Invariants

1. **Orch is single-threaded**: only one thread ever calls `submit_*` or holds
   the `Orchestrator`. No locking is needed on TensorMap, Scope, or Ring-head
   for self-writes.
2. **Tags are consumed at submit**: `task_args.tag(i)` is read only inside
   `submit_*`. Phases after submit (slot storage, dispatch, execution) do not
   see tags.
3. **Slot is parent-heap**: all `TaskSlotState` state is in the parent
   process's heap. Child processes (PROCESS mode workers) never read slot
   state; they receive task data via the mailbox (see
   [worker-manager.md](worker-manager.md) §4).
4. **Ring.alloc is the only blocking point in Orch**: `submit_*` never
   blocks except on ring pressure.
5. **Scope.register_task is idempotent per slot per scope level**: each
   submitted slot gets exactly one scope ref at its current scope depth.

---

## 10. Related

- [distributed_level_runtime.md](distributed_level_runtime.md) — how
  Orchestrator fits alongside Scheduler and Worker
- [scheduler.md](scheduler.md) — what happens to slots after they're pushed
  onto the wiring queue
- [task-flow.md](task-flow.md) — the data (Callable / TaskArgs / CallConfig)
  being moved by `submit_*`
