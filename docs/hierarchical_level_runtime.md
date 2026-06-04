# Hierarchical Level Runtime — Level Model and Component Composition

Callable identity update: public hierarchical registration returns
`CallableHandle`; local IPC task frames carry the handle hash digest and each
target resolves it to a private local slot. See
[callable-identity-registration.md](callable-identity-registration.md).

This document covers:

- The **L0–L6 level model** (what each level represents)
- The **three engine components** (Orchestrator / Scheduler / Worker) and
  their division of responsibility
- How components compose recursively from L3 upward

For details of each component's internals, see:

- [orchestrator.md](orchestrator.md) — submit flow, TensorMap, Scope, Ring, task state machine
- [scheduler.md](scheduler.md) — dispatch loop, queues, completion handling
- [worker-manager.md](worker-manager.md) — WorkerThread pool, fork + mailbox
- [task-flow.md](task-flow.md) — Callable / TaskArgs / CallConfig data flow, execution leaves
- [remote-l3-worker-design.md](remote-l3-worker-design.md) — design proposal
  for scheduling remote L3 workers as NEXT_LEVEL children

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
  (fork + shm for today's L3 and for local recursive L4+ composition).
  Cross-host L4/L5/L6 composition, where a parent schedules a remote L3
  endpoint over RoCE/HCCS/UB/sockets, is a proposed extension.

| Level | Workers it contains | Status |
| ----- | ------------------- | ------ |
| L3 (Host) | `ChipWorker` ×N + `SubWorker` ×M | Implemented |
| L4 (Pod) | `Worker(level=3)` ×N + `SubWorker` ×M | Local-only implemented; remote proposed |
| L5 (SuperNode) | `Worker(level=4)` ×N | Local L4 code path, untested; remote proposed |
| L6 (Cluster) | `Worker(level=5)` ×N | Local L4 code path, untested; remote proposed |

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
(next-level pool and sub pool). Each `WorkerThread` owns one std::thread that
encodes `(callable, config, args_blob)` into a `MAILBOX_SIZE`-byte shared
memory region, signals the pre-forked Python child, and spin-polls
`TASK_DONE`.

- Next-level (chip) children run `_chip_process_loop`, which constructs a
  `ChipWorker` and dispatches each kernel through it.
- SUB children run `_sub_worker_loop`, which decodes the args blob into a
  `TaskArgs` and calls the registered Python callable as `fn(args)`. There
  is no C++ `SubWorker` class — SUB workers exist only as a worker-type
  enum value plus a Python child loop.

See [worker-manager.md](worker-manager.md) for the dispatch state machine,
fork ordering, and mailbox layout. See [task-flow.md](task-flow.md) for
what flows through `ChipWorker::run`.

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
                 │                                 │                              encode mailbox → spin-poll TASK_DONE
                 │                                 │                              (blocking; child runs the kernel)
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
| WorkerThread ↔ child | shm mailbox (state + error + task data) | encoded blob |
| Python ↔ C++ | nanobind bindings | TaskArgs / CallConfig / callable handle |
| Tensor data | `torch.share_memory_()` or host malloc | zero-copy shared address |

---

## 4. Recursive Composition

A higher-level `Worker` can register a lower-level `Worker` as a
NEXT_LEVEL child through the same mailbox protocol L3 uses for chip
children. The Python `Worker.add_worker(child)` stores an un-init'd child
Worker; on first `run()`, the parent forks a child process that inits the
inner Worker and enters a mailbox-polling loop (`_child_worker_loop`).

```python
# L3 child: sub-only (or with chips via device_ids)
l3 = Worker(level=3, num_sub_workers=1)
l3_sub_handle = l3.register(lambda: verify_result())

def my_l3_orch(orch, args, config):
    orch.submit_sub(l3_sub_handle)

# L4 parent
w4 = Worker(level=4, num_sub_workers=0)
l3_handle = w4.register(my_l3_orch)
w4.add_worker(l3)
w4.init()

def my_l4_orch(orch, args, config):
    orch.submit_next_level(l3_handle, TaskArgs(), CallConfig())

w4.run(my_l4_orch)
w4.close()
```

When L4's `WorkerThread` writes a task frame to the L3 child's mailbox, the
frame carries the callable hash digest plus `config` and `args_blob`. The child
loop reads the digest, resolves it through its local identity table to a private
orch-function slot, and calls `inner_worker.run(orch_fn, args, cfg)`. The inner
Worker opens its own scope, executes the orch function with its own
`Orchestrator`, and drains. Each level's orch fn receives its own Orchestrator
— recursion is symmetric.

**Nested fork ordering**: L3's own children (sub/chip) are forked **inside**
the L4 child process, on L3's first `run()`. This keeps the process tree
clean: L4 parent → L3 child → L3's sub/chip grandchildren.

**Mode per level is independent**: L3 can use PROCESS (chip children), while
L4 also uses PROCESS (L3 Worker children). Each `Worker` picks its children's
mode independently. Nested forks are safe because L3 init happens inside the
already-forked L3 child process.

See [task-flow.md](task-flow.md) §9 for the full recursive data-flow
walk-through.

---

## 5. Python/C++ Division

| Concern | Python layer | C++ layer |
| ------- | ------------ | --------- |
| Process lifecycle | fork() timing, `SharedMemory` alloc/unlink, waitpid | — |
| Callable registration | owns handle/hashid registries and child-local Python dispatch mappings | — |
| Orchestration DAG | user's orch fn, `submit_*` calls | `Orchestrator::submit_*` engine |
| Scheduling | — | `Scheduler` thread, queues, `WorkerThread` pool |
| Dispatch | — | `WorkerThread::dispatch_process`, mailbox IPC |
| Runtime execution | — | `ChipWorker` via dlsym'd runtime `.so` |

Python handles **when** things happen (fork ordering, lifecycle). C++ handles
**how fast** (threading, atomics, zero-copy dispatch).

---

## 6. Process Model

```text
┌──────────────────────────────────────────────────────────────┐
│  Parent (main) process                                       │
│                                                              │
│  Python main thread (Orch)                                   │
│    │                                                         │
│    ├── C++ Scheduler thread                                  │
│    ├── C++ WorkerThread[0] ── shm mailbox ──► chip child 0   │
│    ├── C++ WorkerThread[1] ── shm mailbox ──► chip child 1   │
│    ├── C++ WorkerThread[2] ── shm mailbox ──► sub  child 0   │
│    └── C++ WorkerThread[3] ── shm mailbox ──► sub  child 1   │
│                                                              │
└─────────────────────────────┬────────────────────────────────┘
                              │ fork() (before any C++ thread starts)
            ┌─────────────────┼─────────────────┐
            ▼                                   ▼
   ┌─────────────────┐                 ┌─────────────────┐
   │ Chip child 0    │                 │ Chip child 1    │
   │ poll mailbox    │       …         │ poll mailbox    │
   │ ChipWorker.run  │                 │ ChipWorker.run  │
   └─────────────────┘                 └─────────────────┘
```

**Fork ordering invariant**: Python forks every child process FIRST, before
any C++ `Scheduler` / `WorkerThread` is started. This avoids the classical
fork-in-a-multi-threaded-process hazard.

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
| `src/common/hierarchical/orchestrator.{h,cpp}` | `Orchestrator`: submit, TensorMap, Scope |
| `src/common/hierarchical/scheduler.{h,cpp}` | `Scheduler`: dispatch loop + queues |
| `src/common/hierarchical/worker_manager.{h,cpp}` | `WorkerManager` + `WorkerThread`: pool, mailbox-IPC dispatch |
| `src/common/hierarchical/worker.{h,cpp}` | `Worker` (L3+): composes the above |
| `src/common/hierarchical/ring.{h,cpp}` | slot allocator |
| `src/common/hierarchical/tensormap.{h,cpp}` | base_ptr → producer slot |
| `src/common/hierarchical/scope.{h,cpp}` | scope lifetime management |
| `src/common/worker/chip_worker.{h,cpp}` | L2 `ChipWorker` (kernel-running leaf, runs in the forked chip child) |
| `python/bindings/` | nanobind exposure of C++ engine to Python |
| `python/simpler/worker.py` | Python `Worker` factory + lifecycle wrapper |
