# Task Flow — Callable / TaskArgs / CallConfig Pass-Through

This document specifies **what data flows through the hierarchical runtime and
what shapes it takes at each stage**. It covers:

- The three handles carried through every level: `Callable`, `TaskArgs`, `CallConfig`
- The `IWorker` interface and its three implementations
- The L2 ABI edge where internal formats are converted to `ChipStorageTaskArgs`
- Recursive composition for L4+
- A single end-to-end walkthrough

For the components that move this data (how it's stored, dispatched,
scheduled), see:

- [orchestrator.md](orchestrator.md) — submit flow, Ring, TensorMap, Scope
- [scheduler.md](scheduler.md) — dispatch loop, queues, completion handling
- [worker-manager.md](worker-manager.md) — WorkerThread, THREAD/PROCESS
  modes, mailbox mechanics
- [distributed_level_runtime.md](distributed_level_runtime.md) — level model
  and how components compose

---

## 1. The three handles

Every task flowing through any level carries exactly three pieces of data:

| Handle | Type | What it is |
| ------ | ---- | ---------- |
| `Callable` | `uint64_t` (opaque) | What the target worker should execute — interpretation depends on the receiving `IWorker` subclass |
| `TaskArgs` | user builder class | Tensors + scalars + per-tensor tags (IN/OUT/INOUT/etc.) |
| `CallConfig` | small POD | Execution knobs (block_dim, aicpu_thread_num, enable_profiling, …) |

Everything else in the engine is either plumbing (slots, ring, tensormap,
scheduler) or per-kernel state (stored in `Callable`).

---

## 2. `Callable` — one type, three meanings

```cpp
using Callable = uint64_t;
```

Opaque 64-bit handle. What it actually is depends on the destination worker:

| Context | `Callable` encodes | Who casts it | How |
| ------- | ------------------ | ------------ | --- |
| `w3.submit_next_level(cb, …)` dispatched to `ChipWorker` (L2) | `ChipCallable*` — C++ object with compiled kernels | `ChipWorker::run` | `reinterpret_cast<ChipCallable*>(callable)` |
| `w4.submit_next_level(cb, …)` dispatched to `Worker(level=3)` (L3 as L4 child) | `OrchFn` — Python orchestration function pointer | `Worker::run` | `reinterpret_cast<OrchFn>(callable)` |
| `w3.submit_sub(cb, …)` dispatched to `SubWorker` | `uint64_t` callable_id indexing `py_registry_` | `SubWorker::run` | direct use as integer |

Where `OrchFn` is:

```cpp
using OrchFn = void (*)(Orchestrator*, TaskArgsView, const CallConfig&);
```

The `submit_*` API uses `Callable` (uint64) uniformly — no `void*` / `int32_t`
split, no three-way cast.

### Lifetime — pre-fork registration

Every concrete `Callable` object (ChipCallable, Python orch fn, sub callable)
**must be registered before any child process is forked**. After fork, the
child inherits these through COW and the uint64 handle dereferences validly
in the child.

---

## 3. `TaskArgs` — one class, four representations

One user-facing class. Its contents appear in four different physical
representations across a task's lifetime — these are **phases**, not
hierarchy levels.

```cpp
class TaskArgs {
    std::vector<ContinuousTensor> tensors_;
    std::vector<TensorArgType>    tags_;     // per-tensor: INPUT/OUTPUT/INOUT/OUTPUT_EXISTING/NO_DEP
    std::vector<uint64_t>         scalars_;
public:
    void add_tensor(const ContinuousTensor&, TensorArgType tag = TensorArgType::INPUT);
    void add_scalar(uint64_t);
    TaskArgsView view() const;
    int32_t tensor_count() const;
    int32_t scalar_count() const;
    TensorArgType tag(int32_t i) const;    // only Orchestrator reads tags
};
```

`TensorArgType` has five values (matches existing `tensor_arg.h:53-59`):
`INPUT`, `OUTPUT`, `INOUT`, `OUTPUT_EXISTING`, `NO_DEP`.

### Representation at each phase

| Phase | Form | Backing memory | Who writes | Who reads |
| ----- | ---- | -------------- | ---------- | --------- |
| **① User submit** | `TaskArgs` object (builder) | Python/C++ parent heap | user orch fn | Orchestrator |
| **② Slot storage** | `TaskArgs` object (inside `slot.task_args`) | parent heap | Orchestrator.submit moves it here | WorkerThread at dispatch |
| **③ Dispatch wire (PROCESS only)** | length-prefixed blob | shm mailbox (MAP_SHARED) | parent WorkerThread encodes | forked child decodes |
| **④ L2 ABI edge** | `ChipStorageTaskArgs` POD (1672 B) | child stack | `ChipWorker::run` assembles | `pto2_run_runtime` consumes |

### Tags stripped at submit

Tags are consumed by `Orchestrator::submit_*` to derive TensorMap dependencies
and then discarded. Phases ②, ③, ④ do not carry tags — scheduler, worker
thread, child, and runtime.so all ignore per-tensor direction.

### Blob byte layout (phase ③)

```text
offset 0:            int32  tensor_count = T
offset 4:            int32  scalar_count = S
offset 8:            ContinuousTensor tensors[T]    // 40 B each
offset 8 + 40T:      uint64_t scalars[S]            // 8 B each
total used:          8 + 40T + 8S
```

No tags, no pickle, no schema versioning — pure memcpy.

### TaskArgsView — the interface type

Both THREAD mode (from `TaskArgs::view()`) and PROCESS mode (from `read_blob`)
yield the same view type:

```cpp
struct TaskArgsView {
    int32_t tensor_count;
    int32_t scalar_count;
    const ContinuousTensor *tensors;   // T items
    const uint64_t         *scalars;   // S items
};
```

24 bytes, POD, passable by value. Where the pointed-to arrays live depends on
mode:

- **THREAD**: `tensors` points into the `std::vector<ContinuousTensor>` heap
  backing inside `slot.task_args`
- **PROCESS**: `tensors` points into the shm mailbox blob region

View does **not** own memory. Valid for the duration of a single `IWorker::run`
call.

### Conversion diagram

```text
① TaskArgs (user)                    — parent heap (vectors)
     │
     │ Orchestrator::submit_next_level (tags consumed)
     ▼
② slot.task_args: TaskArgs           — parent heap, stored in slot

     ┌── THREAD mode ─────────────────────────────────────┐
     │  view = slot.task_args.view()                       │
     │    (pointers into slot's vector backing)            │
     └─────────────────────────────────────────────────────┘
                     OR
     ┌── PROCESS mode ────────────────────────────────────┐
     │  write_blob(mailbox, slot.task_args)                │
     │    (memcpy into shm mailbox)                        │
     │  child reads mailbox:                               │
     │  view = read_blob(mailbox_bytes)                    │
     │    (pointers into shm mailbox)                      │
     └─────────────────────────────────────────────────────┘

     │ (both paths yield TaskArgsView)
     ▼
    IWorker::run(callable, view, config)

     │ (ChipWorker only, at L2 ABI)
     ▼
④ ChipStorageTaskArgs POD — child stack
     │ memcpy view.tensors, view.scalars into struct
     ▼
    pto2_run_runtime(callable, &chip_storage, &config)
```

---

## 4. `CallConfig` — small POD, always by value

```cpp
struct CallConfig {
    int32_t block_dim = 1;
    int32_t aicpu_thread_num = 3;
    bool    enable_profiling = false;
    // future fields here — same POD used at all levels
};
```

Propagated by value throughout:

1. User builds `CallConfig` and passes into `submit_next_level`
2. Orchestrator stores it inline in `slot.config` (POD copy)
3. Dispatch: THREAD passes `const slot.config &`; PROCESS memcpy into mailbox
4. Child reads `CallConfig` from mailbox by value
5. `IWorker::run` receives `const CallConfig&`; passed on to `pto2_run_runtime`
   at the L2 edge

Same type at every level. `ChipCallConfig` is an alias for `CallConfig` at the
L2 runtime ABI (they must have identical layout).

---

## 5. `IWorker` — the unified execution interface

```cpp
class IWorker {
public:
    virtual ~IWorker() = default;
    virtual void run(Callable callable,
                     TaskArgsView args,
                     const CallConfig &config) = 0;
};
```

Three implementations:

### `ChipWorker` (L2 leaf)

Wraps a dlsym'd `runtime.so`. `run()` assembles `ChipStorageTaskArgs` from the
view and calls `pto2_run_runtime`:

```cpp
void ChipWorker::run(Callable cb, TaskArgsView view, const CallConfig &config) override {
    ChipStorageTaskArgs chip_storage;
    chip_storage.tensor_count_ = view.tensor_count;
    chip_storage.scalar_count_ = view.scalar_count;
    memcpy(chip_storage.tensors_, view.tensors, view.tensor_count * sizeof(ContinuousTensor));
    memcpy(chip_storage.scalars_, view.scalars, view.scalar_count * sizeof(uint64_t));
    pto2_run_runtime(reinterpret_cast<ChipCallable*>(cb), &chip_storage, &config);
}
```

~1.6 KB memcpy per task; negligible.

### `SubWorker` (Python callable leaf)

```cpp
void SubWorker::run(Callable cb, TaskArgsView view, const CallConfig &config) override {
    uint64_t cid = cb;                     // no cast
    py_registry_[cid](view);               // invoke Python callable with view
}
```

Child inherits the Python registry through fork COW; the registry lookup works
with no IPC.

### `Worker` (L3+ composite)

Runs one DAG per `run` invocation. The `Callable` is the user's orch fn:

```cpp
void Worker::run(Callable cb, TaskArgsView args, const CallConfig &config) override {
    orchestrator_.scope_begin();
    reinterpret_cast<OrchFn>(cb)(&orchestrator_, args, config);   // user orch fn
    orchestrator_.drain();
    orchestrator_.scope_end();
}
```

User convenience overload:

```cpp
void Worker::run(const Task &task) {
    run(reinterpret_cast<Callable>(task.orch), task.task_args.view(), task.config);
}
```

---

## 6. Data flow through a submit

The user's orch fn receives an `Orchestrator*` (not a `Worker*`) and calls
`submit_next_level` / `submit_sub`:

```cpp
class Orchestrator {
public:
    SubmitResult submit_next_level(Callable cb, TaskArgs args, const CallConfig &config);
    SubmitResult submit_next_level_group(Callable cb, std::vector<TaskArgs> args_list, const CallConfig &config);
    SubmitResult submit_sub(Callable cb, TaskArgs args, const CallConfig &config);
};

struct SubmitResult { TaskSlot slot_id; };
```

Only `slot_id` is returned — downstream consumers reference tensors by their
own pointers (already registered in TensorMap by the OUTPUT/INOUT tag).

Where the data goes after submit:

1. `Callable` — copied into `slot.callable` (parent heap, one `uint64_t`)
2. `TaskArgs` — moved into `slot.task_args` (parent heap, vector-backed).
   Tags are consumed during the same submit call for dep inference and
   **never carried further**.
3. `CallConfig` — copied into `slot.config` (parent heap, POD)

For the full submit mechanics (ring alloc, TensorMap lookup/insert, scope ref,
fanout wiring), see [orchestrator.md](orchestrator.md).

## 7. Data flow through dispatch

After the scheduler picks an idle `WorkerThread` and calls `wt->dispatch(sid)`,
the WorkerThread reads task data from the slot and hands it to
`IWorker::run`:

### THREAD mode — zero-copy

`TaskArgs::view()` returns pointers into the slot's vector backing. No encode,
no memcpy beyond `CallConfig` value-passing.

```cpp
worker_->run(slot.callable, slot.task_args.view(), slot.config);
```

### PROCESS mode — encode once to mailbox

Parent-side WorkerThread encodes callable + config + TaskArgs blob into a
shm mailbox; child reads the blob back as a view:

```text
slot.callable   ─┐
slot.config     ─┼─► memcpy into shm mailbox ─► child reads view ─► worker_->run(cb, view, config)
slot.task_args  ─┘    (write_blob)                (read_blob)
```

The mailbox layout, fork ordering, and child loop are in
[worker-manager.md](worker-manager.md) §4.

### Memory partitioning

| Region | Lives in | Used by | Lifetime |
| ------ | -------- | ------- | -------- |
| `slots_[N]` (TaskSlotState array) | parent heap | Orchestrator, Scheduler, WorkerThread parent side | ring-managed |
| `slots_[i].task_args` (vector-backed) | parent heap | same | until slot released |
| per-WT mailbox (PROCESS only) | shm MAP_SHARED | parent WorkerThread writes, child reads | lifetime of WorkerThread |
| tensor data bytes | torch shm (`share_memory_()` or equiv) | kernel reads/writes | user-managed |
| `Callable` target (ChipCallable / OrchFn / Python fn) | parent heap | child via fork COW | pre-fork registered |

**Child never reads the slot.** Child only sees:

1. its mailbox (shm)
2. parent's pre-fork heap via COW (read-only in practice)
3. MAP_SHARED tensor data buffers

## 8. Data flow on completion

When `IWorker::run` returns, the WorkerThread signals completion:

- **THREAD mode**: direct call to `on_complete_(slot_id)`, which pushes to
  `Scheduler::completion_queue_`
- **PROCESS mode**: child writes `TASK_DONE` to mailbox; parent WorkerThread
  sees it, calls `on_complete_(slot_id)`

At this point:

- Tensor output data is already written to shm (kernel wrote via
  `ContinuousTensor.data` pointer → shm page visible to parent)
- Control returns to the Scheduler, which releases fanout refs and wakes
  downstream consumers

For the completion-side mechanics (fanout release, `try_consume`, ring
release), see [scheduler.md](scheduler.md) §6.

---

## 9. Recursive composition (L4+)

`Worker` implements `IWorker`, so a higher-level `Worker` can use it as a
next-level child:

```python
w3 = Worker(level=3, child_mode=WorkerManager.Mode.PROCESS)
w3.add_worker(NEXT_LEVEL, chip_worker_0)
w3.add_worker(NEXT_LEVEL, chip_worker_1)
w3.add_worker(SUB, sub_worker_0)

w4 = Worker(level=4, child_mode=WorkerManager.Mode.THREAD)
w4.add_worker(NEXT_LEVEL, w3)                 # L3 Worker is IWorker
```

At L4 the `Callable` passed to `submit_next_level` is an **L3 orch fn**, not a
`ChipCallable`:

```python
def my_l3_orch(orch3, args, cfg):
    orch3.submit_next_level(chip_callable_handle, args, cfg)

def my_l4_orch(orch4, args, cfg):
    orch4.submit_next_level(my_l3_orch_handle, args, cfg)

w4.run(Task(orch=my_l4_orch, task_args=..., config=...))
```

L4's WorkerThread dispatches to `w3` via `IWorker::run`. `Worker::run`
interprets the `Callable` as an `OrchFn`:

```cpp
void Worker::run(Callable cb, TaskArgsView args, const CallConfig &cfg) override {
    orchestrator_.scope_begin();
    reinterpret_cast<OrchFn>(cb)(&orchestrator_, args, cfg);
    orchestrator_.drain();
    orchestrator_.scope_end();
}
```

Each level's orch fn receives **its own** `Orchestrator` — the recursion is
symmetric. `Worker` code does not branch on `level_`; the level is only a
label (for diagnostics).

---

## 10. Worked example — one L3 chip task

User code:

```python
a = torch.randn(N).share_memory_()
b = torch.randn(N).share_memory_()
c = torch.zeros(N).share_memory_()

args = TaskArgs()
args.add_tensor(make_ct(a), IN)
args.add_tensor(make_ct(b), IN)
args.add_tensor(make_ct(c), OUT)

def my_orch(orch, view, cfg):
    chip_args = TaskArgs()
    for i in range(view.tensor_count):
        chip_args.add_tensor(view.tensors[i], IN if i < 2 else OUT)
    orch.submit_next_level(chip_kernel_handle, chip_args, cfg)

w3 = Worker(level=3, child_mode=PROCESS)
w3.add_worker(NEXT_LEVEL, chip_worker_0)
w3.init()    # fork chip_0 here

w3.run(Task(orch=my_orch, task_args=args, config=CallConfig(block_dim=3)))
```

Step-by-step (PROCESS mode, one chip worker):

| Step | Where | What happens |
| ---- | ----- | ------------ |
| 1 | parent Python | user builds `args: TaskArgs`, calls `w3.run(Task)` |
| 2 | `Worker::run` | `scope_begin` → call `my_orch(&orch_, args.view(), cfg)` |
| 3 | `Orchestrator::submit_next_level` | `slot = ring.alloc()`; move `chip_args` into `slot.task_args`; walk tags → `tensormap.lookup(a.data)`, `tensormap.lookup(b.data)`, `tensormap.insert(c.data, slot)`; push ready |
| 4 | Scheduler thread | pop `slot`; `wt = manager.pick_idle(NEXT_LEVEL)` (WT_chip_0); `wt->dispatch(slot)` |
| 5 | WT_chip_0 parent side | encode mailbox: write `callable` = chip_kernel handle, `config`, `write_blob` of task_args; set `TASK_READY`; spin-poll |
| 6 | chip_0 child process | wake on `TASK_READY`; `read_blob` → `view`; call `ChipWorker::run(cb, view, cfg)` |
| 7 | `ChipWorker::run` | assemble `ChipStorageTaskArgs` POD (memcpy view); call `pto2_run_runtime(cb, &chip_storage, &cfg)` |
| 8 | runtime.so | translate host ptrs → device ptrs; dispatch AICPU / AICore; write output into `c`'s shm |
| 9 | chip_0 child | `run` returns; write `TASK_DONE` |
| 10 | WT_chip_0 parent | see `TASK_DONE`; call `on_complete_(slot)` |
| 11 | Scheduler | mark slot COMPLETED; fanout release (none in this DAG); scope_end will release scope ref |
| 12 | `Worker::run` returns | user's `w3.run(Task)` returns; `c` contains result in shm, visible to user |

---

## 11. Design notes

### Why `Callable = uint64_t`, not `void*`

All three callable meanings (ChipCallable pointer, OrchFn pointer, sub
callable_id) fit in 64 bits. Using `void*` forced `int32_t callable_id` to go
through `reinterpret_cast<intptr_t>` then `static_cast<int32_t>` — three layers
of cast. `uint64_t` lets each receiver do a single cast appropriate to its
semantics.

### Why tags live only on user-side `TaskArgs`

Tags (IN/OUT/INOUT/…) are used by `Orchestrator::submit_*` to derive TensorMap
dependencies and nothing else. Scheduler, WorkerThread, child, runtime.so, and
kernels do not inspect them. Keeping tags only in Layer ① simplifies the blob
and makes the "tags are Orchestrator input" rule explicit. Matches existing
runtime: `ChipStorageTaskArgs` (`task_args.h:157`) is already declared with
`void` as the TensorTag parameter.

### Why no `WorkerPayload` wrapper

`IWorker::run` takes `(Callable, TaskArgsView, const CallConfig&)` directly.
Wrapping them in a struct added no value and made mailbox serialization
indirect. Task identity (slot_id) is held by the WorkerThread for the
completion callback, not passed into the IWorker.

### Why slots on heap, mailbox on shm

Slots carry scheduler-only state (atomics, mutex, `std::vector` of fanout
consumers) that is parent-private. Putting them in shm would force cross-
process atomics and shm-safe containers. The only data that needs to cross
the fork boundary is per-task: callable, config, args — and that fits in a
~2 KB mailbox with a one-time memcpy per dispatch (matches the pattern
already used by `DistChipProcess` today).

### Why TaskArgs in slot (not encoded blob in slot)

`TaskArgs` is vector-backed. Storing an `uint8_t args_blob[N]` inline in the
slot would cap task size per level and waste memory per slot. Since the slot
is parent-heap, there is no fork-boundary constraint on what it holds — just
store the `TaskArgs` object and encode only at dispatch (PROCESS only), or
hand over `task_args.view()` (THREAD).

### Why `TaskArgsView` is just pointers + counts

View is constructed at both ends of dispatch (from `TaskArgs::view()` and
from `read_blob()`). Making it POD (24 B) lets it pass by value through
`IWorker::run`. The underlying `ContinuousTensor[]` lives either in the
vector's heap backing or inline in the mailbox blob — view doesn't care.

---

## Related

- [distributed_level_runtime.md](distributed_level_runtime.md) — L0–L6 level
  model, three-component composition
- [orchestrator.md](orchestrator.md) — how `submit_*` actually builds the DAG
- [scheduler.md](scheduler.md) — how dispatched slots get worker threads
- [worker-manager.md](worker-manager.md) — `WorkerThread`, THREAD/PROCESS
  modes, mailbox layout, fork ordering
- [chip-level-arch.md](chip-level-arch.md) — L2 single-chip: three-program
  model (host / AICPU / AICore)
- [`../src/common/task_interface/task_args.h`](../src/common/task_interface/task_args.h)
  — `TaskArgs` template and `ChipStorageTaskArgs` alias
- [`../src/common/task_interface/tensor_arg.h`](../src/common/task_interface/tensor_arg.h)
  — `ContinuousTensor` POD and `TensorArgType` enum
