# Distributed Level Runtime — Level Model and Component Composition

This document covers:

- The **L0–L6 level model** (what each level represents)
- The **three engine components** (Orchestrator / Scheduler / Worker) and
  their division of responsibility
- How components compose recursively from L3 upward

For details of each component's internals, see:

- [orchestrator.md](orchestrator.md) — submit flow, TensorMap, Scope, Ring, task state machine
- [scheduler.md](scheduler.md) — dispatch loop, queues, completion handling
- [worker-manager.md](worker-manager.md) — WorkerThread pool, THREAD/PROCESS modes, fork + mailbox
- [task-flow.md](task-flow.md) — Callable / TaskArgs / CallConfig data flow, IWorker interface

For the L2 chip-level details (host `.so`, AICPU, AICore), see
[chip-level-arch.md](chip-level-arch.md).

---

## 1. Level Model

The runtime uses a 7-level hierarchy mirroring the physical topology of Ascend
NPU clusters:

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

- **L0–L2** (on-device): AICPU scheduler, AICore/AIV workers, device Global
  Memory. Managed by the chip-level runtime (see
  [chip-level-arch.md](chip-level-arch.md)). Communication via shared GM with
  atomics and barriers.
- **L3–L6** (host/cluster): each level runs the same scheduling engine
  composed of Orchestrator + Scheduler + Worker pool. Communication via IPC
  (fork + shm at L3 today; RDMA / sockets at L4+).

| Level | Workers it contains | Status |
| ----- | ------------------- | ------ |
| L3 (Host) | `ChipWorker` ×N + `SubWorker` ×M | Implemented |
| L4 (Pod) | `Worker(level=3)` ×N | Planned |
| L5 (SuperNode) | `Worker(level=4)` ×N | Planned |
| L6 (Cluster) | `Worker(level=5)` ×N | Planned |

`Worker` is a single C++ class that handles every level from L3 upward — the
`level` parameter is a diagnostic label; behavior does not branch on it. The
same Orchestrator/Scheduler/Worker code runs unchanged.

---

## 2. Three Components — Roles

Every level L3+ runs three cooperating components. Each has its own dedicated
thread in the parent process.

### Orchestrator (Orch thread)

The **DAG builder**. Exposed to the user's orchestration function as the
first argument of `submit_*`. Runs single-threaded on the user's thread.

Owns:

- `Ring` — fixed-size slot pool, allocates with back-pressure
- `TensorMap` — `tensor_base_ptr → producer_slot` lookup, drives automatic dep inference
- `Scope` — lifetime management for intermediate tensors

One `submit_next_level(callable, task_args, config)` call:

1. allocates a slot
2. moves task data into the slot
3. walks `TaskArgs` tags (INPUT/OUTPUT/INOUT/OUTPUT_EXISTING/NO_DEP) to
   lookup/insert TensorMap entries
4. records fanin metadata on producer slots
5. pushes the new slot onto the scheduler's wiring queue

See [orchestrator.md](orchestrator.md) for the 7-step submit flow and state machine.

### Scheduler (Scheduler thread)

The **DAG executor**. A dedicated C++ thread that drains three queues:

- **wiring queue** — slots just submitted; wire fanout edges, compute readiness
- **ready queue** — slots with all fanin satisfied; pick an idle WorkerThread and dispatch
- **completion queue** — slots whose worker finished; release fanout, wake downstream consumers, retire slot

The Scheduler never inspects task data — it just moves slot ids between queues
and consults TaskSlotState metadata.

See [scheduler.md](scheduler.md) for the dispatch loop and coordination.

### Worker / WorkerManager / WorkerThread

The **execution layer**. `WorkerManager` holds two pools of `WorkerThread`s
(next-level pool and sub pool). Each `WorkerThread`:

- owns one `IWorker` (`ChipWorker`, `SubWorker`, or nested `Worker`)
- has its own `std::thread`
- runs in one of two modes:
  - `THREAD`: calls `worker->run(callable, view, config)` directly in-process
  - `PROCESS`: forks a child at init; each dispatch writes task data to a shm
    mailbox, the child decodes and runs

See [worker-manager.md](worker-manager.md) for thread/process mechanics, fork
ordering, and mailbox layout. See [task-flow.md](task-flow.md) for what flows
through `IWorker::run`.

---

## 3. Component Coordination

```text
                   Orch thread                    Scheduler thread             Worker threads
                   ───────────                    ────────────────             ──────────────
  User code ──► Orchestrator                      Scheduler
                 │                                 │
                 │ submit(callable, args, config)  │
                 │   1. ring.alloc()               │
                 │   2. TensorMap lookup/insert    │
                 │   3. record fanin              │
                 │   4. push wiring_queue ───────►│
                 │                                 │ Phase 0: drain wiring_queue
                 │                                 │   wire fanout edges
                 │                                 │   if ready → ready_queue
                 │                                 │ pop ready_queue
                 │                                 │ pick idle WorkerThread
                 │                                 │ wt.dispatch(slot_id) ──────► WorkerThread
                 │                                 │                              worker->run(callable, view, config)
                 │                                 │                              (blocking, THREAD or PROCESS mode)
                 │                                 │◄── completion_queue ────── on_complete_(slot_id)
                 │                                 │ on_task_complete:
                 │                                 │   fanout release
                 │                                 │   wake downstream
                 │                                 │   try_consume → ring release
                 │ drain() ◄── notify when all done │
```

Communication channels:

| Path | Mechanism | Payload |
| ---- | --------- | ------- |
| Orch → Scheduler | wiring_queue (mutex + CV) | slot id |
| Scheduler → WorkerThread | WorkerThread internal queue | slot id |
| WorkerThread → Scheduler | completion_queue (mutex + CV) | slot id |
| WorkerThread ↔ child (PROCESS mode) | shm mailbox (state + error + task data) | encoded blob |
| Python ↔ C++ | nanobind bindings | TaskArgs / CallConfig / callable handle |
| Tensor data | `torch.share_memory_()` or host malloc | zero-copy shared address |

---

## 4. Recursive Composition

A `Worker` is itself an `IWorker`, so a higher-level `Worker` can register it
as a next-level child:

```python
w3 = Worker(level=3, child_mode=WorkerManager.Mode.PROCESS)
w3.add_worker(NEXT_LEVEL, chip_worker_0)
w3.add_worker(SUB,        sub_worker_0)
w3.init()

w4 = Worker(level=4, child_mode=WorkerManager.Mode.THREAD)
w4.add_worker(NEXT_LEVEL, w3)         # w3 is an IWorker
w4.init()

w4.run(Task(orch=my_l4_orch, task_args=..., config=...))
```

When L4's `WorkerThread` dispatches to L3, L3's `Worker::run` opens its own
scope, executes the L4-supplied orch function with L3's own `Orchestrator`,
and drains. Each level's orch fn receives its own Orchestrator — recursion is
symmetric.

**Mode per level is independent**: L3 might use PROCESS (sim isolation), L4
THREAD (L3 workers are thread-safe composites). Each `Worker` chooses its
children's mode at construction.

See [task-flow.md](task-flow.md) §9 for the full recursive data-flow
walk-through.

---

## 5. Python/C++ Division

| Concern | Python layer | C++ layer |
| ------- | ------------ | --------- |
| Process lifecycle | fork() timing, `SharedMemory` alloc/unlink, waitpid | — |
| Callable registration | maintains `py_registry[cid]` for sub callables | — |
| Orchestration DAG | user's orch fn, `submit_*` calls | `Orchestrator::submit_*` engine |
| Scheduling | — | `Scheduler` thread, queues, `WorkerThread` pool |
| Dispatch | — | `WorkerManager::dispatch`, THREAD/PROCESS mode |
| Runtime execution | — | `ChipWorker` via dlsym'd runtime `.so` |

Python handles **when** things happen (fork ordering, lifecycle). C++ handles
**how fast** (threading, atomics, zero-copy dispatch).

---

## 6. Process Model

```text
┌─────────────────────────────────────────────────────┐
│  Parent (main) process                              │
│                                                      │
│  Python main thread (Orch)                           │
│    │                                                 │
│    ├── C++ Scheduler thread                          │
│    ├── C++ WorkerThread[0] → ChipWorker[0]           │
│    ├── C++ WorkerThread[1] → ChipWorker[1]           │
│    ├── C++ WorkerThread[2] → SubWorker[0]            │
│    └── C++ WorkerThread[3] → SubWorker[1]            │
│                                                      │
└──────────────────────────┬──────────────────────────┘
                           │ fork() (PROCESS mode only, before C++ threads start)
            ┌──────────────┼──────────────┐
            ▼                             ▼
   ┌────────────────┐            ┌────────────────┐
   │ Child process 0 │            │ Child process 1 │
   │ poll mailbox    │            │ poll mailbox    │
   │ run callable    │            │ run callable    │
   └────────────────┘            └────────────────┘
```

**Fork ordering invariant**: Python forks child processes FIRST (before C++
`Scheduler` / `WorkerThread` threads are started). This avoids
fork-in-multithreaded-process hazards.

THREAD mode workers skip the fork entirely — they run in the parent process's
`WorkerThread::std::thread`.

---

## 7. Runtime Isolation (Onboard Hardware)

A single device can only run **one runtime** per CANN process context. CANN's
AICPU framework (`libaicpu_extend_kernels.so`) caches the user AICPU `.so` on
first load and skips reloading on subsequent launches. If a different
runtime's AICPU `.so` is launched on the same device, the cached (stale)
function pointers are used, causing hangs.

**Do not reuse a device across different runtimes within a single process.**
Use separate processes (one per runtime), or partition devices so each
runtime gets exclusive devices. See
[testing.md](testing.md#runtime-isolation-constraint-onboard) for the pytest
device allocation algorithm.

---

## 8. Source layout

| Path | Role |
| ---- | ---- |
| `src/common/distributed/orchestrator.{h,cpp}` | `Orchestrator`: submit, TensorMap, Scope |
| `src/common/distributed/scheduler.{h,cpp}` | `Scheduler`: dispatch loop + queues |
| `src/common/distributed/worker_manager.{h,cpp}` | `WorkerManager`: WorkerThread pool |
| `src/common/distributed/worker_thread.{h,cpp}` | `WorkerThread`: THREAD/PROCESS dispatch |
| `src/common/distributed/worker.{h,cpp}` | `Worker` (L3+): composes the above |
| `src/common/distributed/ring.{h,cpp}` | slot allocator |
| `src/common/distributed/tensormap.{h,cpp}` | base_ptr → producer slot |
| `src/common/distributed/scope.{h,cpp}` | scope lifetime management |
| `src/common/worker/chip_worker.{h,cpp}` | L2 `ChipWorker` (IWorker leaf) |
| `src/common/worker/sub_worker.{h,cpp}` | `SubWorker` (IWorker leaf for Python callables) |
| `python/bindings/` | nanobind exposure of C++ engine to Python |
| `python/simpler/worker.py` | Python `Worker` factory + lifecycle wrapper |

> Current source still uses `Dist*` names in several places; rename is tracked
> as a separate cleanup.
