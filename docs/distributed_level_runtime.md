# Distributed Level Runtime

## 1. Level Model

The runtime uses a 7-level hierarchy that mirrors the physical topology of Ascend NPU clusters:

```text
L6  CLOS2 / Cluster    ── full cluster (N6 super-nodes)
L5  CLOS1 / SuperNode  ── super-node (N5 pods)
L4  POD   / Pod        ── pod (4 hosts)
L3  HOST  / Node       ── single host machine (16 chips + M SubWorkers)
L2  CHIP  / Processor  ── one NPU chip (shared device memory)
L1  DIE   / L2Cache    ── chip die (hardware-managed)
L0  CORE  / AIV, AIC   ── individual compute core (hardware-managed)
```

**L2 is the boundary** between two worlds:

- **L0–L2** (on-device): AICPU scheduler, AICore/AIV workers, device Global Memory. Managed by the simpler runtime. Communication via shared GM with atomics and barriers (Tier 1).
- **L3–L6** (host/cluster): each level is a separate process. Communication via IPC — Unix sockets, TCP, or RDMA (Tier 3). L3↔L2 uses host-device DMA (Tier 2).

Every level from L3 upward is designed to run the **same scheduling engine** (`DistWorker`). Currently **only L2 and L3 are implemented**. L4–L6 are the intended future extension — the `DistWorker` class implements `IWorker` so it can be nested, but `DistWorker::run()` is a placeholder today.

| Level | Workers it contains | Status |
| ----- | ------------------- | ------ |
| L3 (Host) | ChipWorker ×N + DistSubWorker ×M | Implemented |
| L4 (Pod) | DistWorker(3) ×N (each is an L3 node) | Planned |
| L5 (SuperNode) | DistWorker(4) ×N | Planned |
| L6 (Cluster) | DistWorker(5) ×N | Planned |

A `DistWorker` at any level implements `IWorker`, so a higher level can treat it as just another worker — recursive composition. The scheduling engine, DAG tracking, and scope management are designed to be identical at every level. Today only L3 uses this engine; L4+ will reuse it when inter-node IPC is added.

## 2. One Level: Orchestrator / Scheduler / Worker

Within each level, three roles cooperate:

```text
                    Orch thread                    Scheduler thread             Worker threads
                    ───────────                    ────────────────             ──────────────
User code ──►  DistOrchestrator                   DistScheduler
               │                                   │
               │ submit(callable, args, config)     │
               │   1. alloc ring slot               │
               │   2. TensorMap: lookup deps        │
               │   3. record fanin metadata         │
               │   4. push wiring_queue ───────────►│
               │                                    │ Phase 0: drain wiring_queue
               │                                    │   wire fanout edges (lock + dep_pool)
               │                                    │   if ready → push ready_queue
               │                                    │ pop ready_queue
               │                                    │ pick idle WorkerThread
               │                                    │ dispatch(payload) ──────► IWorker::run()
               │                                    │                           (blocking)
               │                                    │◄── worker_done(slot) ────  return
               │                                    │ on_task_complete:
               │                                    │   fanout release
               │                                    │   wake downstream tasks
               │                                    │   try_consume → ring release
               │                                    │
               │ drain() ◄── notify when all done ──│
```

**Orchestrator** (main thread, single-threaded):

- Owns TensorMap, Scope, Ring alloc side — no locks needed
- Builds the dependency metadata: for each submit, looks up input tensors to find producers, records fanin pointers in payload, increments producers' `fanout_count`
- Pushes tasks to scheduler's wiring queue for asynchronous fanout edge construction

**Scheduler** (dedicated C++ thread):

- **Wiring**: drains wiring queue, wires fanout edges (acquires `fanout_lock`, allocates dep_pool entries), determines task readiness
- Pops ready tasks from ready queue, finds idle WorkerThreads, dispatches
- Receives completion callbacks from WorkerThreads
- Releases fanout refs, wakes downstream consumers, retires consumed slots

**WorkerThread** (one per IWorker, dedicated thread):

- Wraps one `IWorker` (ChipWorker, DistSubWorker, or nested DistWorker)
- Calls `worker->run(payload)` synchronously — blocks until done
- Notifies Scheduler via `worker_done(slot)`

## 3. How It Works: Scope, TensorMap, RingBuffer

### TensorMap — automatic dependency inference

TensorMap maps `tensor_base_ptr → producer_task_slot`. When a task is submitted:

```text
submit(inputs=[ptr_A, ptr_B], outputs=[ptr_C]):

  TensorMap.lookup(ptr_A) → slot 3 (producer)  → fanin edge: 3 → current
  TensorMap.lookup(ptr_B) → not found           → no dependency
  TensorMap.insert(ptr_C, current_slot)          → future consumers will depend on us
```

The user never explicitly declares "task X depends on task Y". Dependencies are inferred from which tasks produce/consume the same tensor addresses.

### RingBuffer — slot allocation with back-pressure

The ring manages a fixed window of task slots (`DIST_TASK_WINDOW_SIZE = 128`). The Orchestrator calls `alloc()` to claim the next slot. If all slots are occupied by in-flight tasks, `alloc()` blocks until a slot is freed — this is **back-pressure**, preventing the Orchestrator from running too far ahead of the Scheduler.

```text
alloc() ──► [slot 0][slot 1]...[slot 127] ──► release()
  ↑ blocks if full                              ↑ called when task CONSUMED
```

### Scope — intermediate tensor lifetime

Scopes group tasks whose intermediate outputs should be released together. Each task submitted inside a scope carries one extra "scope reference" in its fanout count. When `scope_end()` is called, that reference is released for every task in the scope, allowing completed tasks with no downstream consumers to reach CONSUMED and free their ring slot.

```python
with hw.scope():
    r1 = hw.submit(...)   # r1 gets scope ref (fanout_total += 1)
    r2 = hw.submit(...)   # r2 gets scope ref
# scope_end: release scope ref on r1 and r2
# if r1/r2 have no downstream consumers, they transition to CONSUMED
```

Without scopes, tasks with no downstream consumers would never be consumed (no one releases their fanout ref), eventually exhausting the ring.

### Task State Machine

```text
FREE ──► PENDING ──► READY ──► RUNNING ──► COMPLETED ──► CONSUMED
           │           │          │            │              │
         has fanin   fanin=0   Scheduler    worker(s)     all fanout
         deps        satisfied  dispatches   done          refs released
                                                          → ring slot freed
```

For group tasks, RUNNING → COMPLETED requires ALL N workers to finish (`sub_complete_count == group_size`).

## 4. Python/C++ Division and Process/Thread Model

### Division of Responsibility

```text
Python layer                              C++ layer
──────────────                            ──────────────
Worker                                    DistWorker
  - fork() SubWorker processes              - DistOrchestrator (DAG, TensorMap)
  - register callables (before fork)        - DistScheduler (thread, dispatch)
  - manage SharedMemory lifecycle           - DistRing (slot allocation)
  - provide submit() / scope() API         - WorkerThread (per-worker thread)
  - call drain() to wait                    - DistSubWorker (mailbox I/O)
                                            - ChipWorker (device runtime)
```

Python handles **process lifecycle** (fork, waitpid, SharedMemory alloc/unlink). C++ handles **scheduling and execution** (threads, atomics, condition variables).

### Process Model

```text
┌─────────────────────────────────────────────────────┐
│  Main process                                        │
│                                                      │
│  Python main thread (Orch)                           │
│    │                                                 │
│    ├── C++ Scheduler thread                          │
│    ├── C++ WorkerThread[0] → ChipWorker[0]           │
│    ├── C++ WorkerThread[1] → ChipWorker[1]           │
│    ├── C++ WorkerThread[2] → DistSubWorker[0]        │
│    └── C++ WorkerThread[3] → DistSubWorker[1]        │
│                                                      │
└──────────────────────────┬───────────────────────────┘
                           │ fork() (before C++ threads start)
            ┌──────────────┼──────────────┐
            ▼                             ▼
   ┌────────────────┐            ┌────────────────┐
   │ Child process 0 │            │ Child process 1 │
   │ Python loop:    │            │ Python loop:    │
   │  poll mailbox   │            │  poll mailbox   │
   │  run callable   │            │  run callable   │
   └────────────────┘            └────────────────┘
```

**Fork ordering**: Python forks child processes FIRST, then creates C++ threads (`DistWorker.init()`). This avoids POSIX fork-in-multithreaded-process issues.

### Data Exchange

| Path | Mechanism | Data |
| ---- | --------- | ---- |
| Orch → Scheduler | `DistReadyQueue` (mutex + CV) | task slot index |
| Scheduler → WorkerThread | `WorkerThread.queue_` (mutex + CV) | `WorkerPayload` copy |
| WorkerThread → Scheduler | `completion_queue_` (mutex + CV) | task slot index |
| WorkerThread ↔ Child process | SharedMemory mailbox (256 bytes, acquire/release) | callable_id, state, error_code |
| Python ↔ ChipWorker | `WorkerPayload.callable` / `.args` (raw pointers) | ChipCallable buffer, TaskArgs |
| All tensors | `torch.share_memory_()` or host malloc | zero-copy shared address space |

## 5. Unified Interface — Designed for All Levels

The API is designed so that orchestration functions can be reused across levels without modification — only the physical workers change. Currently L2 and L3 are implemented; L4+ will use the same `submit()` / `scope()` / `drain()` interface.

### Core Operations

```python
# At any level:
worker.submit(worker_type, payload, inputs=[...], outputs=[...])  # submit a task
worker.submit(..., args_list=[a0, a1, a2, a3])                    # submit a group task
with worker.scope():                                               # scope lifetime
    worker.submit(...)
worker.run(Task(orch=my_orch))                                    # run and drain
```

### L2 Usage — Single Chip

```python
w = Worker(level=2, device_id=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")
w.init()
w.run(chip_callable, chip_args, block_dim=24)
w.close()
```

### L3 Usage — Multiple Chips + SubWorkers

```python
w = Worker(level=3, device_ids=[0, 1], num_sub_workers=2,
           platform="a2a3sim", runtime="tensormap_and_ringbuffer")
cid = w.register(my_python_fn)     # register before init (inherited by fork)
w.init()

def my_orch(w, args):
    # Build callable and task args (same types as L2)
    chip_callable = ChipCallable.build(signature, func_name, orch_bin, children)
    task_args = ChipStorageTaskArgs()
    task_args.add_tensor(make_tensor_arg(input_tensor))
    task_args.add_tensor(make_tensor_arg(output_tensor))

    with w.scope():
        # ChipWorker task: runs kernel on NPU
        payload = WorkerPayload()
        payload.callable = chip_callable.buffer_ptr()
        payload.args = task_args.__ptr__()
        payload.block_dim = 24
        r = w.submit(WorkerType.NEXT_LEVEL, payload, outputs=[64])

        # SubWorker task: runs Python callable, depends on chip output
        sub_p = WorkerPayload()
        sub_p.callable_id = cid
        w.submit(WorkerType.SUB, sub_p, inputs=[r.outputs[0].ptr])

w.run(Task(orch=my_orch))
w.close()
```

### L3 Group Task — N Chips as One Logical Worker

```python
def my_orch(w, args):
    # Each chip gets its own args with rank-specific data
    args_list = []
    for rank in range(4):
        a = ChipStorageTaskArgs()
        a.add_tensor(make_tensor_arg(input))
        a.add_tensor(make_tensor_arg(output))
        a.add_scalar(rank)
        a.add_scalar(4)
        args_list.append(a.__ptr__())

    # 1 DAG node, 4 chips execute in parallel
    w.submit(WorkerType.NEXT_LEVEL, payload, args_list=args_list, outputs=[out_size])
```

### Why It's Uniform

The internal C++ interface is `IWorker::run(payload)` — one method, implemented by every worker type:

| Implementation | What `run()` does |
| -------------- | ----------------- |
| `ChipWorker` | Calls NPU runtime → device executes kernel |
| `DistSubWorker` | Writes shared-memory mailbox → forked child executes Python callable |
| `DistChipProcess` | Writes shared-memory mailbox → forked child runs ChipWorker (process-isolated) |
| `DistWorker` | Placeholder for L4+ — will run sub-engine (Orchestrator + Scheduler + workers) |

The `IWorker` interface enables recursive composition: an L4 `DistWorker` would contain L3 `DistWorker` instances as workers, dispatching to them via `run()`. This is the intended design for L4+, not yet implemented.

## 6. Callable Hierarchy

Callable types form a hierarchy that mirrors the level model. Each level has its own callable type with different storage and transport characteristics:

```text
CoreCallable   ── C++ struct, binary kernel code, resolved by platform loader
     │              Lives on device. Used by AICPU dispatch (resolved_addr_).
     │
ChipCallable   ── C++ struct, orchestration binary + nested CoreCallable children
     │              Serialized binary layout — host builds it, copies to AICPU.
     │              Template: Callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 32>
     │
HostCallable   ── Python object, never leaves host process
                   Contains: ChipCallable(s) + Python orch fn + Python sub fns
                   C++ side stores only callable_id (int) in WorkerPayload.
```

### CoreCallable and ChipCallable (C++ binary layout)

These two types use a C++ `Callable<>` template with fixed-size arrays and flexible array member (FAM) storage. They are **serialization formats** — the host builds them into a `vector<uint8_t>` buffer via `make_callable()`, then the buffer is copied to device memory or passed by pointer.

```cpp
// src/common/task_interface/callable.h
using CoreCallable = Callable<void, CORE_MAX_TENSOR_ARGS, 0>;       // leaf: kernel binary
using ChipCallable = Callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 32>; // parent: orch + children
```

**CoreCallable** carries one kernel binary and a `resolved_addr_` field. The platform loader (`device_runner.cpp`) resolves the binary to a device address and writes `resolved_addr_`. AICPU dispatch reads `resolved_addr()` to jump to the kernel entry point.

**ChipCallable** carries the orchestration binary, a `func_name` for AICPU entry, and up to 32 child CoreCallables indexed by `func_id`. The Python binding (`ChipCallable.build()`) constructs the serialized buffer; the C runtime treats it as an opaque `const void*` pointer in `WorkerPayload.callable`.

### HostCallable (Python-only, L3+)

HostCallable is a **Python-side composition** — it never crosses the host↔device boundary and has no C++ struct representation. It combines:

- **ChipCallable(s)**: pre-compiled binary callables for ChipWorker tasks
- **Python orch function**: the DAG orchestration logic (calls `w.submit()`)
- **Python sub functions**: host-side callables for SubWorker tasks

At L3, the Python orch function runs in the main thread and submits tasks to the C++ scheduling engine. ChipWorker tasks carry `WorkerPayload.callable` (ChipCallable buffer pointer). SubWorker tasks carry `WorkerPayload.callable_id` (int index into a Python registry). The C++ Scheduler does not inspect either — it dispatches the payload to the appropriate WorkerThread.

At L4+, a HostCallable submitted to a higher-level engine would also use `callable_id` in WorkerPayload. The C++ Scheduler treats it as opaque — the receiving DistWorker looks up the ID in its Python registry to retrieve the full HostCallable object.

### Design Rationale

Only CoreCallable and ChipCallable require C++ binary layout because they cross the host↔device boundary (DMA copy to AICPU). HostCallable stays in host memory and is better represented as a Python object — it contains Python functions which cannot be serialized into a C struct.

## Architecture Diagram

```text
Python Application
  │
  └─► Worker                              ← Python wrapper (lifecycle, fork management)
       │
       └── DistWorker(level=3)               ← C++ scheduling engine
            │
            ├── DistOrchestrator             ← submit(), TensorMap, Scope
            ├── DistScheduler                ← ready_queue → WorkerThread dispatch
            ├── DistRing                     ← slot allocator with back-pressure
            ├── DistTensorMap                ← base_ptr → producer slot mapping
            ├── DistScope                    ← scope lifetime management
            │
            ├── ChipWorker ×N               ← IWorker: NPU device execution
            │    └── DeviceRunner (thread_local)
            │
            └── DistSubWorker ×M            ← IWorker: fork/shm Python callable
                 └── forked child process    ← mailbox state machine
```

## Runtime Isolation (Onboard Hardware)

A single device can only run **one runtime** per CANN process context. CANN's AICPU framework (`libaicpu_extend_kernels.so`) caches the user AICPU .so on first load and skips reloading on subsequent launches. If a different runtime's AICPU .so is launched on the same device, the cached (stale) function pointers are used, causing hangs.

This means: **do not reuse a device across different runtimes within a single process.** Either use separate processes (one per runtime), or partition devices so each runtime gets exclusive devices. See [testing.md](testing.md#runtime-isolation-constraint-onboard) for details and the pytest device allocation algorithm.

## Files

| File | Purpose |
| ---- | ------- |
| `src/common/distributed/dist_types.h/.cpp` | WorkerPayload, DistTaskSlotState, IWorker, DistReadyQueue |
| `src/common/distributed/dist_orchestrator.h/.cpp` | submit / submit_group, TensorMap wiring, scope |
| `src/common/distributed/dist_scheduler.h/.cpp` | Scheduler thread, WorkerThread, group dispatch/completion |
| `src/common/distributed/dist_worker.h/.cpp` | Top-level engine: composes all components |
| `src/common/distributed/dist_ring.h/.cpp` | Circular slot allocator with back-pressure |
| `src/common/distributed/dist_tensormap.h/.cpp` | base_ptr → producer slot mapping |
| `src/common/distributed/dist_scope.h/.cpp` | Scope depth tracking and ref management |
| `src/common/distributed/dist_sub_worker.h/.cpp` | fork/shm IWorker with mailbox protocol |
| `src/common/worker/chip_worker.h/.cpp` | L2 device execution, thread_local DeviceRunner |
| `python/worker.py` | Unified Worker (L2 + L3): Python wrapper, fork management, scope context manager |
| `python/bindings/dist_worker_bind.h` | nanobind bindings for distributed types |
