# Task Flow — Callable / TaskArgs / CallConfig Pass-Through

Callable identity update: public Python submit APIs now accept
`CallableHandle` objects returned by `Worker.register`, and hierarchical task
mailboxes carry the handle's 32-byte hash digest. Target-local integer slots
remain private to the receiving worker. Older `cid` references in this document
describe historical or target-local internals; the authoritative contract is
[callable-identity-registration.md](callable-identity-registration.md).

This document specifies **what data flows through the hierarchical runtime and
what shapes it takes at each stage**. It covers:

- The three handles carried through every level: `Callable`, `TaskArgs`, `CallConfig`
- The `ChipWorker::run` execution leaf at L2
- The L2 ABI edge where internal formats are converted to `ChipStorageTaskArgs`
- Recursive composition for L4+
- A single end-to-end walkthrough

For the components that move this data (how it's stored, dispatched,
scheduled), see:

- [orchestrator.md](orchestrator.md) — submit flow, Ring, TensorMap, Scope
- [scheduler.md](scheduler.md) — dispatch loop, queues, completion handling
- [worker-manager.md](worker-manager.md) — WorkerThread, mailbox IPC mechanics
- [hierarchical_level_runtime.md](hierarchical_level_runtime.md) — level model
  and how components compose

---

## 1. The three handles

Every task flowing through any level carries exactly three pieces of data:

| Handle | Type | What it is |
| ------ | ---- | ---------- |
| `CallableHandle` / `CallableIdentity` | hash digest + kind + namespace | What the target worker should execute; targets resolve the digest to a local slot |
| `TaskArgs` | user builder class | Tensors + scalars + per-tensor tags (IN/OUT/INOUT/etc.) |
| `CallConfig` | small POD | Execution knobs (block_dim, aicpu_thread_num, profiling/dump/PMU flags, …) |

Everything else in the engine is either plumbing (slots, ring, tensormap,
scheduler) or target-local executable state resolved from the callable digest.

---

## 2. Callable Identity

```cpp
struct CallableIdentity {
    std::array<uint8_t, 32> digest;
    CallableKind kind;
    TargetNamespace target_namespace;
};
```

Python users submit `CallableHandle` objects returned by `Worker.register`.
The Python facade validates ownership/liveness and passes `CallableIdentity`
to C++:

| Context | Namespace | How it's consumed |
| ------- | --------- | ----------------- |
| `w3.submit_next_level(handle, …)` dispatched to a chip child | `LOCAL_CHIP` | child resolves digest to its private chip slot, then calls `ChipWorker::run(local_slot, …)` |
| `w4.submit_next_level(handle, …)` dispatched to an L3 `Worker` child | `LOCAL_PYTHON` | child resolves digest to an orchestration function and calls `inner_worker.run(orch_fn, …)` |
| remote `w4.submit_next_level(handle, …)` dispatched to remote L3 | `REMOTE_TASK_DISPATCHER` | remote endpoint resolves digest in its dispatcher registry and calls its embedded L3 Worker |
| `w3.submit_sub(handle, …)` dispatched to a SUB child | `LOCAL_PYTHON` | child resolves digest to a Python callable and calls `fn(args)` |

All three paths share one mailbox wire format: `MAILBOX_OFF_CALLABLE` is
reserved, and the 32-byte digest prefixes the args blob. The receiving child
does the digest-to-slot resolve in its own address space.

The proposed remote L3 path keeps the same callable identity contract, but
sends it in a versioned TASK frame. The remote endpoint resolves the digest
against its own registry after it has reported `HELLO READY`.

### Lifetime — materialize before dispatch

Pre-start registration is captured in the startup snapshot inherited by child
processes. Post-start registration uses the local control plane and completes
only after every active target in scope has installed the digest or reported
failure. A task is dispatched only after registration succeeds.

Remote L3 cannot rely on fork-time COW inheritance. Remote callable
registration uses explicit descriptors: required `PYTHON_IMPORT` paths,
optional negotiated PR #839 serialized Python callable payloads, and
`CHIP_CALLABLE` payloads for inner L3 chip work. A remote callable identity
becomes visible only after the selected endpoint replies success.
The current Python surface implements `RemoteCallable("module:qualname")` as
the required `PYTHON_IMPORT` baseline and requires an explicit `workers=[...]`
list naming remote worker ids returned by `add_remote_worker(...)`.

---

## 3. `TaskArgs` — one class, four representations

One user-facing class. Its contents appear in four different physical
representations across a task's lifetime — these are **phases**, not
hierarchy levels.

```cpp
class TaskArgs {
    std::vector<Tensor> tensors_;
    std::vector<TensorArgType>    tags_;     // per-tensor: INPUT/OUTPUT/INOUT/OUTPUT_EXISTING/NO_DEP
    std::vector<uint64_t>         scalars_;
public:
    void add_tensor(const Tensor&, TensorArgType tag = TensorArgType::INPUT);
    void add_scalar(uint64_t);
    TaskArgsView view() const;
    int32_t tensor_count() const;
    int32_t scalar_count() const;
    TensorArgType tag(int32_t i) const;    // only Orchestrator reads tags
};
```

`TensorArgType` has five values (matches existing `tensor.h:45-51`):
`INPUT`, `OUTPUT`, `INOUT`, `OUTPUT_EXISTING`, `NO_DEP`.

For remote L3 submits, public Python still uses the same `TaskArgs` builder.
`TaskArgs.add_tensor(RemoteTensorRef(...), tag)` appends a normal
`Tensor` metadata entry with `data == 0` plus a hidden remote
sidecar at the same tensor index. The local mailbox path rejects non-empty
remote sidecars; the remote framed path encodes the sidecar as a
`RemoteTensorDescWire`.

### Representation at each phase

| Phase | Form | Backing memory | Who writes | Who reads |
| ----- | ---- | -------------- | ---------- | --------- |
| **① User submit** | `TaskArgs` object (builder) | Python/C++ parent heap | user orch fn | Orchestrator |
| **② Slot storage** | `TaskArgs` object (inside `slot.task_args`) | parent heap | Orchestrator.submit moves it here | WorkerThread at dispatch |
| **③ Dispatch wire (PROCESS only)** | length-prefixed blob | shm mailbox (MAP_SHARED) | parent WorkerThread encodes | forked child decodes |
| **④ L2 ABI edge** | `ChipStorageTaskArgs` POD | child stack | `ChipWorker::run` assembles | `pto2_run_runtime` consumes |

### Tags stripped at submit

Tags are consumed by `Orchestrator::submit_*` to derive TensorMap dependencies
and then discarded. Phases ②, ③, ④ do not carry tags — scheduler, worker
thread, child, and runtime.so all ignore per-tensor direction.

### Blob byte layout (phase ③)

```text
offset 0:            int32  tensor_count = T
offset 4:            int32  scalar_count = S
offset 8:            Tensor tensors[T]    // 128 B each
offset 8 + 128T:      uint64_t scalars[S]            // 8 B each
total used:          8 + 128T + 8S
```

No tags, no pickle, no schema versioning — pure memcpy.

### TaskArgsView — the interface type

The parent-side encoder (from `TaskArgs::view()`) and the child-side
decoder (over the mailbox blob bytes) yield the same view type:

```cpp
struct TaskArgsView {
    int32_t tensor_count;
    int32_t scalar_count;
    const Tensor *tensors;   // T items
    const uint64_t         *scalars;   // S items
};
```

24 bytes, POD, passable by value. Where the pointed-to arrays live depends on
mode:

- **THREAD**: `tensors` points into the `std::vector<Tensor>` heap
  backing inside `slot.task_args`
- **PROCESS**: `tensors` points into the shm mailbox blob region

View does **not** own memory. Valid for the duration of a single
`ChipWorker::run` call in the forked child.

### Conversion diagram

```text
① TaskArgs (user)                    — parent heap (vectors)
     │
     │ Orchestrator::submit_next_level (tags consumed)
     ▼
② slot.task_args: TaskArgs           — parent heap, stored in slot
     │
     │ LocalMailboxEndpoint::run: memcpy into shm mailbox blob
     │   layout = [int32 T][int32 S][Tensor × T][uint64 × S]
     ▼
③ shm mailbox bytes (MAP_SHARED)     — visible to forked child
     │
     │ child decodes header → builds TaskArgsView over the blob bytes
     ▼
    child resolves digest -> local slot
    ChipWorker::run(local_slot, view, config)  (in the forked child)

     │ (L2 ABI edge)
     ▼
④ ChipStorageTaskArgs POD — child stack
     │ memcpy view.tensors, view.scalars into struct
     ▼
    pto2_run_runtime(local_slot, &chip_storage, &config)
```

---

## 4. `CallConfig` — small POD, always by value

```cpp
struct CallConfig {
    int32_t block_dim = 0;  // 0 = auto (DeviceRunner resolves to stream max at run() time)
    int32_t aicpu_thread_num = 3;
    int32_t enable_l2_swimlane = 0;  // perf_level 0–4 (0=off, 4=full)
    int32_t enable_dump_args = 0;
    int32_t enable_pmu = 0;           // 0 = disabled; >0 selects PMU event type
    int32_t enable_dep_gen = 0;
    int32_t enable_scope_stats = 0;
    char    output_prefix[1024] = {};
    // future fields here - same POD used at all levels
};
```

Propagated by value throughout:

1. User builds `CallConfig` and passes into `submit_next_level`
2. Orchestrator stores it inline in `slot.config` (POD copy)
3. Dispatch: `LocalMailboxEndpoint::run` memcpys the slot's `CallConfig`
   into the shm mailbox
4. Remote dispatch: `RemoteL3Endpoint::run` encodes the fields into
   `CallConfigWire` instead of memcpying the POD
5. Child reads `CallConfig` from mailbox by value, or the remote session
   runner reconstructs it from `CallConfigWire`
6. `ChipWorker::run` receives `const CallConfig&`; passed on to
   `pto2_run_runtime` at the L2 edge

Same type at every level. Used directly at the L2 runtime ABI.

---

## 5. Execution leaves — what runs the kernel

There is no abstract `IWorker` interface; dispatch ends in one of two
concrete leaves, each consumed by its own Python child loop.

### `ChipWorker` (NEXT_LEVEL, L2 leaf)

Wraps a dlsym'd `runtime.so`. `_chip_process_loop` instantiates one
`ChipWorker` per chip child and calls its `run` on every dispatch.
`run()` assembles a `ChipStorageTaskArgs` POD from the decoded view and
calls `pto2_run_runtime`:

```cpp
void ChipWorker::run(int32_t local_slot, TaskArgsView view, const CallConfig &config) {
    ChipStorageTaskArgs chip_storage;
    chip_storage.tensor_count_ = view.tensor_count;
    chip_storage.scalar_count_ = view.scalar_count;
    memcpy(chip_storage.tensors_, view.tensors, view.tensor_count * sizeof(Tensor));
    memcpy(chip_storage.scalars_, view.scalars, view.scalar_count * sizeof(uint64_t));
    pto2_run_runtime(local_slot, &chip_storage, &config);
}
```

One memcpy of a few KB per task; negligible.

#### TRB temporary buffer

`tensormap_and_ringbuffer` stages ordinary non-child tensor arguments through a
runner-scoped retained temporary buffer instead of a per-run `device_malloc()` /
`device_free()` pair. This is always on for TRB — an internal allocation
optimization with no user-facing switch. It is not serialized in task mailboxes
and does not change `TaskArgs`, `CallConfig`, child-memory tensors, or public
`Worker.malloc()` / `Worker.free()` semantics.

On each TRB bind the host runtime sizes the retained buffer from the run's
non-child tensors, growing it (free old + malloc new) only when a run needs
more than is currently retained, and bump-slices each tensor from it. The
buffer lives on the `DeviceRunner` across runs (freed once at finalize); the
platform only stores its `{addr, size}` slot. If a grow allocation fails the
run fails before device argument staging. See the runtime's `RUNTIME_LOGIC.md`
§2.4 for the grow/reuse mechanics.

### SUB-type child loop (Python callable leaf)

SUB execution is handled entirely in Python. The forked child process
runs `_sub_worker_loop` which reads the args blob from the shared-memory
mailbox, decodes it into a `TaskArgs` object, and passes it to the
registered callable:

```python
fn(args)    # args: TaskArgs decoded from the mailbox blob
```

The callable receives the same `TaskArgs` that was submitted via
`orch.submit_sub(handle, args)`, with tags stripped (tags are consumed by
the Orchestrator at submit time). There is no C++ class for SUB workers
— the Python child loop and callable registry are the entire
implementation; the child inherits the Python registry through fork COW.

### L4+ recursion — no extra leaf type

A higher-level `Worker` is **not** itself an execution leaf. When L4
dispatches to an L3 child, the child process runs `_child_worker_loop`,
which resolves the digest to the registered orch fn and calls
`inner_worker.run(orch_fn, args, config)` — i.e. the L3 `Worker.run`
Python method, not a C++ leaf. The kernel-running leaves stay at L2
(`ChipWorker`); higher levels just compose more scheduling engines. A remote
L3 session runner follows the same execution shape after it has prestarted its
inner L3 Worker, but task/control/completion bytes travel through the remote
framed protocol instead of the local mailbox.

---

## 6. Data flow through a submit

The user's Python orch fn receives an `Orchestrator` facade (not a `Worker`)
and calls `submit_next_level` / `submit_sub`. These Python methods return
`None`; the task slot remains internal to the scheduling engine.

```python
class Orchestrator:
    # NEXT_LEVEL placement is required. For local Python Worker children and
    # remote L3 dispatch, stable ids are returned by add_worker(...) or
    # add_remote_worker(...). For L3 ChipCallable dispatch, worker ids are
    # the existing chip worker ids.
    def submit_next_level(self, handle, args, config=None, *, worker) -> None: ...
    def submit_next_level_group(self, handle, args_list, config=None, *, workers) -> None: ...
    def submit_sub(self, handle, args=None) -> None: ...
    def submit_sub_group(self, handle, args_list) -> None: ...
```

The C++ implementation still allocates an internal task slot to drive
scheduling, but nanobind does not expose that slot. Downstream consumers
reference tensors by their own pointers (already registered in TensorMap by
the OUTPUT/INOUT tag).

Where the data goes after submit:

1. `CallableIdentity` — copied into `slot.callable` (parent heap)
2. `TaskArgs` — moved into `slot.task_args` (parent heap, vector-backed).
   Tags are consumed during the same submit call for dep inference and
   **never carried further**.
3. `CallConfig` — copied into `slot.config` (parent heap, POD)

For the full submit mechanics (ring alloc, TensorMap lookup/insert, scope ref,
fanout wiring), see [orchestrator.md](orchestrator.md).

## 7. Data flow through dispatch

For local endpoints, after the Scheduler resolves the submitted NEXT_LEVEL
target (or chooses an idle SUB worker), `LocalMailboxEndpoint` encodes
`(callable digest, CallConfig, TaskArgs)` into the per-worker shm mailbox and
the forked child decodes it. Remote NEXT_LEVEL dispatch through
`RemoteL3Endpoint` serializes the same logical payload into a framed TASK
request instead.

Local mailbox path:

```text
slot.callable.digest ─┐
slot.config          ─┼─► memcpy into shm mailbox ─► child resolves digest
slot.task_args       ─┘    (dispatch_process)         and runs local slot
```

For SUB children the same mailbox layout is reused; the Python child
runs `_sub_worker_loop`, which decodes the args blob via
`_read_args_from_mailbox` into a `TaskArgs` object and calls
`fn(args)` directly — no C++ leaf involved.

The mailbox layout, fork ordering, and child loop are in
[worker-manager.md](worker-manager.md).

### Memory partitioning

| Region | Lives in | Used by | Lifetime |
| ------ | -------- | ------- | -------- |
| `Ring` slot-state pool (`std::deque<unique_ptr<TaskSlotState>>`) | parent heap | Orchestrator, Scheduler, WorkerThread parent side | monotonic task-id; reset at `Worker.run` drain |
| `slot.task_args` (single) or `task_args_list[N]` (group, vector-backed) | parent heap | same | until slot reaches CONSUMED |
| per-WT mailbox | shm MAP_SHARED | parent WorkerThread writes, child reads | lifetime of WorkerThread |
| **HeapRing[0..3]** (user OUTPUT auto-alloc + `orch.alloc`) | **4 separate shm MAP_SHARED mmaps**, one per scope-layer ring | output to user code; inherited by forked children | per-ring FIFO via `rings_[r].last_alive`; scope depth picks the ring |
| tensor data bytes (user-provided) | torch shm (`share_memory_()` or equiv) | kernel reads/writes | user-managed |
| Registered callables (ChipCallable / orch fn / Python fn) | parent heap | child via fork COW or `CTRL_REGISTER` IPC | pre-fork or dynamically registered |

Slot state lives inside `Ring` as `std::deque<std::unique_ptr<…>>` so
`push_back` never invalidates pointers to live slots.
`ring.slot_state(id)` hands out a stable pointer for every live slot;
`drain()` calls `ring.reset_to_empty()` to drop all slot state at the
end of each `Worker.run`, bounding per-run memory.

The HeapRing is **partitioned into `MAX_RING_DEPTH = 4` independent
rings** (Strict-1; matches L2's `PTO2_MAX_RING_DEPTH`). Each ring is its
own `mmap(MAP_SHARED | MAP_ANONYMOUS)` taken before fork, so children
inherit all four at the same virtual addresses. The `heap_ring_size`
knob on `Worker(...)` is the **per-ring** size (default 1 GiB → 4 GiB
total VA reservation); physical pages remain lazy under
`MAP_ANONYMOUS`. A task's ring is chosen by scope depth,
`min(scope_depth, MAX_RING_DEPTH - 1)`, so inner-scope tasks
reclaim independently of outer-scope tasks. See
[orchestrator.md §5](orchestrator.md) for the allocator internals and
[orchestrator.md §6](orchestrator.md) for the scope → ring mapping.

**Child never reads the slot.** Child only sees:

1. its mailbox (shm)
2. parent's pre-fork heap via COW (read-only in practice)
3. MAP_SHARED tensor data buffers

## 8. Data flow on completion

When the child finishes the kernel, it writes `TASK_DONE` to the mailbox;
`LocalMailboxEndpoint::run` exits its spin-poll, reads the mailbox error
fields, and returns a `WorkerCompletion`. `MAILBOX_OFF_ERROR == 0` maps to
success; a non-zero child error maps to task failure. The parent
`WorkerThread` pushes that completion onto `Scheduler::completion_queue_`.

At this point:

- Tensor output data is already written to shm (kernel wrote via
  `Tensor.data` pointer → shm page visible to parent)
- Control returns to the Scheduler, which marks the slot `COMPLETED` on
  success or `FAILED` on task/endpoint failure, then releases fanout refs and
  either wakes or poisons downstream consumers

For the completion-side mechanics (fanout release, `try_consume`, ring
release), see [scheduler.md](scheduler.md) §6.

---

## 9. Recursive composition (L4+)

A higher-level `Worker` registers a lower-level `Worker` as a
NEXT_LEVEL child via a mailbox just like L3 does for `ChipWorker`. The
parent side is uniform — `WorkerThread` calls the endpoint `run()` contract and
doesn't care what kind of child is on the other end. The local forked
child runs `_child_worker_loop`, which resolves each dispatched digest and
delegates to
`inner_worker.run(...)` — i.e. another full scheduling engine inside.

### Setup

```python
# L3 child: sub-only (no chips for this example)
l3 = Worker(level=3, num_sub_workers=1)
l3_sub_handle = l3.register(lambda: verify_result())

def my_l3_orch(orch, args, config):
    orch.submit_sub(l3_sub_handle)

# L4 parent
w4 = Worker(level=4, num_sub_workers=0)
l3_handle = w4.register(my_l3_orch) # register L3 orch fn in Python dict
l3_worker_id = w4.add_worker(l3)    # add un-init'd L3 Worker as child
w4.init()

def my_l4_orch(orch, args, config):
    orch.submit_next_level(
        l3_handle,
        TaskArgs(),
        CallConfig(),
        worker=l3_worker_id,
    )

w4.run(my_l4_orch)
w4.close()
```

At L4 the handle passed to `submit_next_level` is a `LOCAL_PYTHON` handle
that maps to a Python orchestration function, not a `ChipCallable`.

### Fork sequence

L4's `init()` allocates the L4 Worker's HeapRing (before fork), then eagerly
runs `_start_hierarchical()` — `init()` is the single startup point, and it
returns only once the whole tree is READY:

1. Forks one child process per L3 Worker child
2. **Inside the child**: `inner_worker.init()` is eager and recursive — it
   creates the L3 Worker (mmaps L3's own HeapRing), forks L3's sub/chip
   children, and blocks on their `INIT_READY` before it returns.
3. Child publishes `INIT_READY` (whole L3 subtree ready), then enters
   `_child_worker_loop(mailbox, registry, inner_worker)` for dispatch
4. **Parent**: awaits each child's `INIT_READY`, then registers each mailbox
   with L4's Worker via `add_next_level_worker_at(worker_id, mailbox_addr)`

```text
L4 parent process
  ├─ Worker(4) + HeapRing (MAP_SHARED, inherited by L3 child)
  └─ fork ──────────────────► L3 child process
                                 ├─ inner_worker.init()   (eager, recursive)
                                 │    ├─ Worker(3) + L3's own HeapRing
                                 │    └─ forks L3's sub/chip children, awaits
                                 │       their INIT_READY
                                 ├─ publish INIT_READY (subtree ready)
                                 └─ _child_worker_loop(mbox, registry, inner_worker)
                                      └─ on dispatch: inner_worker.run(orch_fn, args, cfg)
```

### Dispatch walkthrough

| Step | Where | What happens |
| ---- | ----- | ------------ |
| 1 | L4 parent Python | `w4.run(my_l4_orch)` → `scope_begin` → `my_l4_orch(orch4, ...)` |
| 2 | L4 `Orchestrator.submit_next_level` | the L3 callable handle digest is stored in the slot's callable identity; slot pushed to L4's ready queue |
| 3 | L4 Scheduler | pop the target worker's FIFO → that L3 child's mailbox |
| 4 | L4 WorkerThread (PROCESS) | encode `(callable digest, config, args_blob)` into mailbox; write `TASK_READY`; spin-poll |
| 5 | L3 child `_child_worker_loop` | wake on `TASK_READY`; read digest → child-local slot → `my_l3_orch` |
| 6 | L3 child | `inner_worker.run(my_l3_orch, args, cfg)` → `scope_begin` → `my_l3_orch(orch3, ...)` |
| 7 | L3 `Orchestrator.submit_sub` | `l3_sub_handle` digest dispatched to L3's own sub worker child |
| 8 | L3 sub child | child resolves digest to its local Python callable and executes `verify_result()` |
| 9 | L3 drain | all L3 tasks complete; `scope_end` + `drain` return |
| 10 | L3 child | `inner_worker.run()` returns; `_child_worker_loop` writes `TASK_DONE` |
| 11 | L4 LocalMailboxEndpoint | sees `TASK_DONE`; returns success completion |
| 12 | L4 drain | L4 scope_end + drain; `w4.run()` returns |

Each level's orch fn receives **its own** `Orchestrator` — the recursion is
symmetric. `Worker` code does not branch on `level`; the level is only a
diagnostic label.

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
    orch.submit_next_level(chip_kernel_handle, chip_args, cfg, worker=0)

w3 = Worker(level=3, child_mode=PROCESS)
w3.add_worker(NEXT_LEVEL, chip_worker_0)
w3.init()    # fork chip_0 here

w3.run(my_orch, args, CallConfig(block_dim=3))
```

Step-by-step (one chip worker):

| Step | Where | What happens |
| ---- | ----- | ------------ |
| 1 | parent Python | user builds `args: TaskArgs`, calls `w3.run(my_orch, args, config)` |
| 2 | `Worker::run` | `scope_begin` → call `my_orch(&orch_, args.view(), cfg)` |
| 3 | `Orchestrator::submit_next_level` | `slot = ring.alloc()`; move `chip_args` into `slot.task_args`; walk tags → `tensormap.lookup(a.data)`, `tensormap.lookup(b.data)`, `tensormap.insert(c.data, slot)`; push ready |
| 4 | Scheduler thread | pop `slot` from worker 0's FIFO; resolve stable worker ID 0 to WT_chip_0; dispatch |
| 5 | WT_chip_0 parent side | encode mailbox: write reserved callable field, `config`, digest prefix, `write_blob` of task_args; set `TASK_READY`; spin-poll |
| 6 | chip_0 child process | wake on `TASK_READY`; resolve digest to local slot; `read_blob` → `view`; call `ChipWorker::run(local_slot, view, cfg)` |
| 7 | `ChipWorker::run` | assemble `ChipStorageTaskArgs` POD (memcpy view); call `pto2_run_runtime(local_slot, &chip_storage, &cfg)` |
| 8 | runtime.so | translate host ptrs → device ptrs; dispatch AICPU / AICore; write output into `c`'s shm |
| 9 | chip_0 child | `run` returns; write `TASK_DONE` |
| 10 | WT_chip_0 parent | see `TASK_DONE`; push success completion |
| 11 | Scheduler | mark slot COMPLETED; fanout release (none in this DAG); scope_end will release scope ref |
| 12 | `Worker::run` returns | user's `w3.run(...)` returns; `c` contains result in shm, visible to user |

---

## 11. Design notes

### Why `CallableIdentity`, not a raw integer

Parent-side task slots need a stable identity that is valid across child
processes even when each target uses a different private execution slot. The
submitted `CallableIdentity` carries the 32-byte digest plus scheduling
metadata; each child resolves that digest to its own local slot immediately
before execution.

### Why tags live only on user-side `TaskArgs`

Tags (IN/OUT/INOUT/…) are used by `Orchestrator::submit_*` to derive TensorMap
dependencies and nothing else. Scheduler, WorkerThread, child, runtime.so, and
kernels do not inspect them. Keeping tags only in Layer ① simplifies the blob
and makes the "tags are Orchestrator input" rule explicit. Matches existing
runtime: `ChipStorageTaskArgs` (`task_args.h:157`) is already declared with
`void` as the TensorTag parameter.

### Why no `WorkerPayload` wrapper

`ChipWorker::run` takes `(local_slot, TaskArgsView, const CallConfig&)`
directly. Wrapping them in a struct added no value and made mailbox serialization
indirect. Task identity (slot_id) is held by the parent's WorkerThread
for the completion callback, not passed into the child.

### Why slots on heap, mailbox on shm

Slots carry scheduler-only state (atomics, mutex, `std::vector` of fanout
consumers) that is parent-private. Putting them in shm would force cross-
process atomics and shm-safe containers. The only data that needs to cross
the fork boundary is per-task: callable, config, args — and that fits in a
~2 KB mailbox with a one-time memcpy per dispatch.

### Why TaskArgs in slot (not encoded blob in slot)

`TaskArgs` is vector-backed. Storing an `uint8_t args_blob[N]` inline in the
slot would cap task size per level and waste memory per slot. Since the slot
is parent-heap, there is no fork-boundary constraint on what it holds — just
store the `TaskArgs` object and encode it into the mailbox blob at dispatch
time.

### Why `TaskArgsView` is just pointers + counts

View is constructed at both ends of the mailbox handshake (from
`TaskArgs::view()` on the parent side for encoding, from a decoded
mailbox blob on the child side). Making it POD (24 B) lets it pass by
value through `ChipWorker::run`. The underlying `Tensor[]`
lives in the mailbox blob bytes on the child side — view doesn't care.

---

## Related

- [hierarchical_level_runtime.md](hierarchical_level_runtime.md) — L0–L6 level
  model, three-component composition
- [orchestrator.md](orchestrator.md) — how `submit_*` actually builds the DAG
- [scheduler.md](scheduler.md) — how dispatched slots get worker threads
- [worker-manager.md](worker-manager.md) — `WorkerThread`, mailbox
  layout, fork ordering
- [chip-level-arch.md](chip-level-arch.md) — L2 single-chip: three-program
  model (host / AICPU / AICore)
- [`../src/common/task_interface/task_args.h`](../src/common/task_interface/task_args.h)
  — `TaskArgs` template and `ChipStorageTaskArgs` alias
- [`../src/common/task_interface/tensor.h`](../src/common/task_interface/tensor.h)
  — `Tensor` POD and `TensorArgType` enum
