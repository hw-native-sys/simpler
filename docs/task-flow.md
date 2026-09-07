# Task Flow — Callable / TaskArgs / CallConfig Pass-Through

Public Python submit APIs accept
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
- [hierarchical-level-runtime.md](hierarchical-level-runtime.md) — level model
  and how components compose

---

## 1. The three handles

Every task flowing through any level carries exactly three pieces of data:

| Handle | Type | What it is |
| ------ | ---- | ---------- |
| `CallableHandle` / `CallableIdentity` | hash digest + kind + namespace | What the target worker should execute; targets resolve the digest to a local slot |
| `TaskArgs` | user builder class | Self-describing Tensor views + scalars + per-tensor tags (IN/OUT/INOUT/etc.) |
| `CallConfig` | small POD | Execution knobs (aicpu_thread_num, profiling/dump/PMU flags, …) |

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
| `w3.submit_next_level(handle, …)` dispatched to a chip child | `LOCAL_CHIP` | child resolves digest to its private chip slot, then uses blocking `ChipWorker::run` on the compatibility path or the prepare/launch/poll/finalize native lifecycle on a two-frame endpoint |
| `w4.submit_next_level(handle, …)` dispatched to an L3 `Worker` child | `LOCAL_PYTHON` | child resolves digest to an orchestration function and calls `inner_worker.run(orch_fn, …)` |
| remote `w4.submit_next_level(handle, …)` dispatched to remote L3 | `REMOTE_TASK_DISPATCHER` | remote endpoint resolves digest in its dispatcher registry and calls its embedded L3 Worker |
| `w3.submit_sub(handle, …)` dispatched to a SUB child | `LOCAL_PYTHON` | child resolves digest to a Python callable and calls `fn(args)` |

All three paths share the same logical task-frame payload:
`MAILBOX_OFF_CALLABLE` is reserved, the run's generation-safe pipeline lease
follows `CallConfig`, and the 32-byte digest prefixes the args blob. SUB,
nested/remote L3, simulation, A5, and depth-one fallbacks carry that payload in
the base compatibility frame. A direct A2/A3 onboard chip child can instead
use either of two task frames after the base control frame, with
`PREPARE_READY -> FRAME_STAGED -> ACTIVATE` separating endpoint preparation
from native launch. The receiving child resolves the digest in its own address
space in both forms.

The remote L3 path keeps the same callable identity contract, but
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

The canonical C++ builder is declared in
[`task_args_wire.h`](../src/common/task_interface/task_args_wire.h):

```cpp
using TaskArgs = TaskArgsTpl<Tensor, uint64_t, 0, 0, TensorArgType>;
```

It stores vectors of `Tensor` records, scalar values, tensor tags, and
`ExplicitTaskDependency` records. `add_dep` adds a retaining dependency;
`add_dep_wait` adds an ordering-only dependency. Tensors must be added before
scalars. `TensorArgType` has five values: `INPUT`, `OUTPUT`, `INOUT`,
`OUTPUT_EXISTING`, and `NO_DEP`.

Each `Tensor` embeds its `BufferDescriptor` and a strided view. It carries no
materialized address. `ChipTensor` is the device descriptor produced only at
the L2 boundary; it is not the element stored in user-facing `TaskArgs`.
See [Buffer Memory Model](buffer-abi.md) for identity, access, and lifetime.

For remote L3 submits, `TaskArgs.add_tensor(RemoteTensorRef(...), tag)`
creates a `Tensor` with a `REMOTE_SIDECAR` backend and keeps the remote
reference in a sidecar at the same tensor index. The local mailbox path rejects
remote sidecars; remote dispatch encodes the reference in its framed protocol.

### Representation at each phase

| Phase | Form | Backing memory | Who writes | Who reads |
| ----- | ---- | -------------- | ---------- | --------- |
| **① User submit** | `TaskArgs` builder with `Tensor` records | parent heap | user orchestration | Orchestrator |
| **② Slot storage** | `TaskArgs` inside the task slot | parent heap | Orchestrator | endpoint at dispatch |
| **③ Dispatch wire (PROCESS only)** | counts, `Tensor` records, scalars | shm mailbox | endpoint blob encoder | child decoder and ImportRegistry |
| **④ L2 ABI edge** | `ChipStorageTaskArgs` containing `ChipTensor` records | child-owned materialization | child materializer | native chip runtime |

### Tags and explicit dependencies stay parent-side

Tags drive TensorMap dependency inference, and explicit handles add
host-graph task edges. Neither is serialized into the dispatch blob or copied
into the L2 argument POD.

### Blob byte layout (phase ③)

```text
offset 0:            int32 tensor_count = T
offset 4:            int32 scalar_count = S
offset 8:            Tensor tensors[T]       // 144 bytes each
offset 8 + 144T:     uint64_t scalars[S]      // 8 bytes each
total used:          8 + 144T + 8S
```

`task_args_blob_size` and `write_blob` use `sizeof(Tensor)`, whose 144-byte
size is guarded in [`buffer.h`](../src/common/task_interface/buffer.h).
`read_blob` validates the counts against the available bytes. The blob has no
outer schema version; each tensor contains a descriptor validated on access.

### TaskArgsView — the interface type

`read_blob` returns a non-owning view over the encoded bytes:

```cpp
struct TaskArgsView {
    int32_t tensor_count;
    int32_t scalar_count;
    const uint8_t *tensor_bytes;
    const uint64_t *scalars;

    Tensor tensors(int32_t i) const;
};
```

On a 64-bit host the view is 24 bytes. `tensors(i)` bounds-checks the index,
copies the record to aligned local storage, and validates its descriptor and
view. The tensor region starts after the 8-byte blob header, so readers must
not reinterpret it as an aligned `Tensor*`. The referenced storage must remain
alive while the view is consumed; a staged endpoint owns an immutable frame
snapshot for that purpose.

### Conversion diagram

```text
① TaskArgs (Tensor views, scalars, tags, explicit dependencies)
     │ Orchestrator consumes tags and explicit dependencies
     ▼
② slot.task_args (parent heap)
     │ write_blob: [int32 T][int32 S][Tensor × T][uint64 × S]
     ▼
③ mailbox bytes → child-owned args
     │ ImportRegistry resolves each descriptor to a consumer-local base
     │ materialize_task_args builds the chip representation
     ▼
④ ChipStorageTaskArgs (ChipTensor records + scalars)
     │ native run or prepare/launch/poll/finalize lifecycle
     ▼
    chip runtime stages host-backed data or uses owned device memory
```

A public L2 `Worker.run()` performs the same materialization in its own
process, without a mailbox. It accepts `TaskArgs`, not `ChipStorageTaskArgs`.

---

## 4. `CallConfig` — small POD, always by value

```cpp
struct CallConfig {
    int32_t aicpu_thread_num = 0;  // auto
    int32_t enable_chip_swimlane = 0;  // perf_level 0–4 (0=off, 4=full)
    int32_t enable_dump_args = 0;
    int32_t enable_pmu = 0;           // 0 = disabled; >0 selects PMU event type
    int32_t enable_dep_gen = 0;
    int32_t enable_scope_stats = 0;
    int32_t capture_clock_anchors = 0; // set by the ChipWorker child, not by callers
    RuntimeEnv runtime_env;      // three arrays of four uint64_t overrides
    char    output_prefix[1024] = {};
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
   `simpler_run` at the L2 edge

Same type at every level. Used directly at the L2 runtime ABI.

---

## 5. Execution leaves — what runs the kernel

There is no abstract `IWorker` interface; dispatch ends in one of two
concrete leaves, each consumed by its own Python child loop.

### `ChipWorker` (NEXT_LEVEL, L2 leaf)

The Python chip child resolves the callable digest to a local slot, decodes
`TaskArgs`, and resolves each `Tensor` through its `ImportRegistry`.
`materialize_task_args` builds `ChipStorageTaskArgs` at the resolved bases.
The child then calls the native `run_materialized` binding, or submits a native
run token for the staged prepare/launch/poll/finalize path.

This boundary is a descriptor resolution and materialization step, not a
memcpy from the mailbox tensor array into `ChipTensor[]`. Host-backed arguments
may need device staging and output copy-back; device-backed arguments must
resolve to allocations owned by the target chip. The native `ChipWorker`
consumes the resulting POD and invokes the runtime's execution lifecycle.

#### Pipeline resource leases

A2/A3 host runtimes and the A5 tensor-map-and-ring-buffer runtime declare
`pipeline_depth = 2` as resource capacity. The contract determines the
concrete copy count rather than permitting concurrent device execution by
itself:

| Resource class | Copies | Selection |
| -------------- | -----: | --------- |
| `HOST_PER_RUN` | `pipeline_depth` | lease `slot_id` |
| `DEVICE_SCRATCH` | 1 | slot 0 |
| `EXEC_HANDLE` | `pipeline_depth` | lease `slot_id`, plus any hardware-generation key |

A run-owned lease is `{slot_id, generation}`. `PipelineSlotPool` mints it and
is the authority on ownership: releasing the current lease is idempotent, while
releasing it after the slot has been re-leased is rejected.

`ChipWorker` is downstream of that pool and cannot re-derive ownership — it
never sees an acquire or a release. It keeps a per-slot high-water mark and
rejects any generation below it, which stops a *superseded* lease from
selecting resources the slot's newer owner holds. That is strictly weaker than
an ownership check: a lease that was released but whose successor has not yet
reached `ChipWorker` still passes, because nothing has raised the mark. Closing
that window needs the admission layer to gate dispatch on `pool.owns(lease)`
before handing work down, which is whole-run admission's job, not this
layer's.

Run streams are outside that lease, and there is one pair of them per runner
rather than one per slot. A slot indexes the resources *preparation* mutates,
and preparing a run writes nothing to a stream: only launch submits, and launch
holds the exclusive execution claim, so runs reach the device one at a time and
the stream orders them. The two streams stay distinct because the AICPU Run
kernel spins in the handshake waiting for the AICore workers — one queue would
leave the AICore submission behind a spin that never ends.

The AICore stream carries instruction-cache state. Registered callables keep
content-hash deduplicated GM allocations: simultaneously resident code images
occupy different allocations, while unregister frees an allocation that a later
registration may reuse. A dedup miss is the only repeatable path that publishes
new AICore instruction bytes; after that H2D copy succeeds the pair is marked
stale, and the next launch destroys the AICore stream and creates a replacement.
(The `kernel_entry` ELF that `rtRegisterAllKernel` publishes needs no such mark:
CANN offers no unregister, so it is registered once per runner and released with
the streams at finalize.) Without a new publication the pair stays warm even
when two resident callables alternate. Unproven completion still destroys the
AICore stream conservatively, and only the run that submitted the pair may
retire it — a prepared successor overlaps its predecessor's execution and must
leave the live pair alone. The AICPU stream carries no instruction cache state
and lives for the runner.

#### Whole-run FIFO admission

L3 graph callbacks remain synchronous and serialized, and how many live run
reservations native admission allows is the depth the child backends
negotiated — not a constant. At the negotiated depth two:

```text
run N:     EXECUTING
run N+1:   PREPARED
run N+2:   blocked in begin_run before its graph callback
```

Where a backend publishes depth one, the *second* submission is the one that
blocks in `begin_run`, and there is no prepared successor at all. Code that
relies on a later callback to unblock an earlier run deadlocks on such a
backend.

A5 tensor-map-and-ring-buffer publishes depth two for whole-run admission, but
its local endpoint still has one mailbox frame and device execution remains
serial. The second reservation may build and become `PREPARED`; it does not
gain a second concurrently executable device frame. A5 host-build-graph
publishes no contract and stays at depth one.

`begin_run` acquires a generation-safe lease before invoking the callback.
The FIFO head may enter `EXECUTING` while its callback is still building, which
preserves orchestration callbacks that submit device work and wait for L2
communication before returning. A non-head run remains `BUILDING` or becomes
`PREPARED` when graph construction closes; it cannot execute until every prior
run is terminal. The scheduler observes only the ready-queue partition belonging
to that single active FIFO head. A run's device effects therefore cannot
interleave with another run even when both graphs contain ready tasks. TensorMap
keys remain `(run_id, tensor_key)`, so adjacent runs may reuse the same tensor
address without creating cross-run dependencies.

The terminal transition releases the reservation and lease exactly once,
wakes whichever submission was blocked on capacity, and activates the next prepared run. Empty
runs take the same transition immediately. If graph construction fails, every
unstarted slot is poisoned and consumed, its ready-queue partition is erased,
and the lease is returned without dispatching device work.

Each direct chip child publishes its runtime contract's `pipeline_depth` in
the startup mailbox before `INIT_READY`. The parent configures admission to the
minimum published depth. Backends without a depth-two contract therefore keep
depth-one serial behavior instead of receiving an invalid slot-1 lease.

Whole-run admission decides when a slot may be leased and carries the lease
from `TaskSlot` through the chip mailbox into the runtime slot, so a production
run executes under the lease its run holds rather than unconditionally on slot
0. The scheduler dispatches device work only for the run that holds the FIFO
head and still owns its lease.

The L2 host-runtime boundary exposes `prepare -> launch -> poll/wait ->
finalize`, and the existing `simpler_run` / `ChipWorker.run` surface is the
blocking composition of those phases. `prepare` constructs and binds the
per-run `Runtime` without crossing the device launch fence; `launch` returns
after the backend has actually submitted its execution; `finalize` owns
validation, copy-back, DFX, and Runtime destruction. A backend that advertises
concurrent native preparation may own one active token and one prepared but
unlaunched and unaccepted successor token in separate lease-selected banks.
Other backends permit only one unfinished native run, including a
prepared-but-not-launched run.

#### Two-frame endpoint staging lane

A direct A2/A3 chip endpoint with a negotiated depth of at least two uses two
task frames and advertises `supports_frame_staging`. One `WorkerThread` owns
both frames and drives them through a non-blocking progress interface; the
child process likewise has one loop that services control traffic, both task
frames, and the bounded active/prepared native lifecycles. There is no thread
per frame.

The active and successor paths are:

```text
IDLE -> TASK_READY    -> FRAME_STAGED -> TASK_LAUNCHED -> TASK_DONE | TASK_FAILED
IDLE -> PREPARE_READY -> FRAME_STAGED -> ACTIVATE -> TASK_LAUNCHED
                                               -> TASK_DONE | TASK_FAILED
```

`FRAME_STAGED` means that the child owns an immutable frame snapshot; it does
not by itself distinguish validation-only staging from completed native
preparation. For a `host_build_graph` successor whose own configuration and
active predecessor are both non-diagnostic, the child constructs a
generation-bound native run in the leased inactive arena bank while the
predecessor executes. If the predecessor is already active, that preparation
finishes before `FRAME_STAGED` publication. A successor that reaches the child
before any predecessor owns the active claim publishes validation-only, so the
parent can activate it without deadlock; native prepare follows activation or
a later predecessor claim without another mailbox state transition. HBG tasks
adjacent to diagnostic state and all
`tensormap_and_ringbuffer` tasks also use this as a validation-only state:
native prepare waits for the predecessor's complete device fence because their
shared diagnostic or device-scratch state is not safe to rewrite early.
Backend and per-run capabilities, rather than the mailbox protocol, select
between these meanings.

An HBG successor's prepared token remains unlaunched and unaccepted until
`ACTIVATE`, and activation still cannot launch it until the predecessor has
polled complete and finalized. The sticky acceptance word therefore remains
zero throughout preparation. Shutdown, stale activation, and pre-launch
failure finalize the token exactly once before the frame becomes terminal.

The scheduler stages only the first eligible single NEXT_LEVEL task from the
prepared FIFO successor. Tasks from the active run use only the active lane, so
the second frame cannot create same-device execution overlap. Prepared groups
remain on their normal queue and dispatch synchronously after FIFO promotion.
Remote, SUB, A5, simulation, nested-worker, and single-frame endpoints retain
the blocking compatibility path.

The child validates newly visible metadata from both frames before selecting
the next active `dispatch_id`. It prepares that active token first; preparation
of an ordinary HBG successor starts only after the selected predecessor owns
the native active claim. Physical frame order therefore cannot reorder native
prepare or launch.

Every task frame carries protocol, run, lease slot, generation, dispatch, and
callable identity. Its sticky acceptance word is separate from the state word
and is set only by the real native launch marker. `FRAME_STAGED` never satisfies
the launch fence. A terminal pre-launch failure may conservatively retire the
run-level acceptance waiter, but it does not set the frame's acceptance word.
The parent clears that word only immediately before reusing an `IDLE` frame.
Control commands continue to use a separate base frame, so they cannot
overwrite either staged task frame. Callable prepare/register/unregister
commands defer while an active or backend-prepared token owns runtime state.
The default unbounded control wait therefore follows child liveness through a
long run; a finite control timeout includes this deferral interval, and expiry
poisons the local endpoint because the pending command's completion is
uncertain.

#### TRB temporary buffer

`tensormap_and_ringbuffer` stages ordinary non-child tensor arguments through a
retained temporary buffer owned per pipeline slot, instead of a per-run `device_malloc()` /
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

SUB execution is handled in Python. `_sub_worker_loop` resolves the callable
digest and uses `ImportRegistry.mapped_args_from_blob` to create a `MappedArgs`
object whose tensor data is mapped into the child's address space:

```python
fn(args)    # args: MappedArgs with mapped tensors and scalars
```

The submitted tensor descriptors and scalars cross the mailbox; dependency
tags stay parent-side. Pre-start registrations enter the fork snapshot;
post-start registrations reach the live child through the control plane.

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
and calls `submit_next_level` / `submit_sub`. NEXT_LEVEL submits return an
opaque, run-scoped `TaskHandle`; SUB submits return `None`.

```python
class Orchestrator:
    # NEXT_LEVEL placement is required. For local Python Worker children and
    # remote L3 dispatch, stable ids are returned by add_worker(...) or
    # add_remote_worker(...). For L3 ChipCallable dispatch, worker ids are
    # the existing chip worker ids.
    def submit_next_level(self, handle, args, config=None, *, worker) -> TaskHandle: ...
    def submit_next_level_group(self, handle, args_list, config=None, *, workers) -> TaskHandle: ...
    def submit_sub(self, handle, args=None) -> None: ...
    def submit_sub_group(self, handle, args_list) -> None: ...
```

The handle exposes no slot fields or constructor in Python. It can only be
passed to `TaskArgs.add_dep` or `TaskArgs.add_dep_wait`, which creates a task
edge without inventing a tensor dependency. `add_dep` retains the producer
until the consumer completes; `add_dep_wait` only waits for producer
completion. Handles from another run are rejected before a slot is allocated.

Where the data goes after submit:

1. `CallableIdentity` — copied into `slot.callable` (parent heap)
2. `TaskArgs` — moved into `slot.task_args` (parent heap, vector-backed).
   Tags and explicit handles are consumed during the same submit call for dep
   inference and **never carried across the dispatch boundary**.
3. `CallConfig` — copied into `slot.config` (parent heap, POD)
4. `PipelineSlotLease` — copied from the owning run into
   `slot.pipeline_lease`; local chip mailboxes forward `{slot_id, generation}`
   to `ChipWorker::run_with_lease`.

For the full submit mechanics (ring alloc, TensorMap lookup/insert, scope ref,
fanout wiring), see [orchestrator.md](orchestrator.md).

## 7. Data flow through dispatch

For local endpoints, after the Scheduler resolves the submitted NEXT_LEVEL
target (or chooses an idle SUB worker), `LocalMailboxEndpoint` encodes
`(callable digest, CallConfig, PipelineSlotLease, TaskArgs)` into the
per-worker shm mailbox and
the forked child decodes it. Remote NEXT_LEVEL dispatch through
`RemoteL3Endpoint` serializes the same logical payload into a framed TASK
request instead.

Every dispatched group member contributes one run-acceptance obligation. For
an A2A3 onboard chip endpoint, the child-side native runner writes
`TASK_ACCEPTED` after its AICore and AICPU kernels are both enqueued; the parent
observes it without releasing the mailbox. Other endpoint paths satisfy the
same obligation conservatively when their completion returns. Acceptance is the
launch fence for that run's own dispatches; it does not admit the next graph
callback. Once the prior callback returns, `begin_run` may invoke the next
serialized callback whenever a negotiated pipeline lease is free, even while
the prior run remains below its acceptance or completion fence.

Local mailbox path:

```text
slot.callable.digest ─┐
slot.config          ─┼─► memcpy into shm mailbox ─► child resolves digest
slot.pipeline_lease  ─┤    (submit_progress)         and runs local slot
slot.task_args       ─┘
```

For SUB children the same mailbox layout is reused; the Python child
runs `_sub_worker_loop`, which decodes the args blob via
`ImportRegistry.mapped_args_from_blob` into a `MappedArgs` object — every
tensor mapped into this process, the scalars alongside — and calls
`fn(args)` directly — no C++ leaf involved. A nested next-level child
runs `_child_worker_loop` instead, which re-exports rather than maps
(`_reexport_args_from_mailbox`) and hands its orch function a `TaskArgs`.

The mailbox layout, fork ordering, and child loop are in
[worker-manager.md](worker-manager.md).

### Memory partitioning

| Region | Lives in | Used by | Lifetime |
| ------ | -------- | ------- | -------- |
| `Ring` slot-state pool (`std::deque<unique_ptr<TaskSlotState>>`) | parent heap | Orchestrator, Scheduler, WorkerThread parent side | monotonic task-id; compacted only when globally quiescent |
| `slot.task_args` (single) or `task_args_list[N]` (group, vector-backed) | parent heap | same | until slot reaches CONSUMED |
| per-WT mailbox | shm MAP_SHARED | parent WorkerThread writes, child reads | lifetime of WorkerThread |
| **HeapRing[0..3]** (user OUTPUT auto-alloc + `orch.alloc`) | **4 separate shm MAP_SHARED mmaps**, one per scope-layer ring | output to user code; inherited by forked children | per-ring FIFO via `rings_[r].last_alive`; scope depth picks the ring |
| tensor data bytes (user-provided) | torch shm (`share_memory_()` or equiv) | kernel reads/writes | user-managed |
| Registered callables (ChipCallable / orch fn / Python fn) | parent heap | child via fork COW or `CTRL_REGISTER` IPC | pre-fork or dynamically registered |

Slot state lives inside `Ring` as `std::deque<std::unique_ptr<…>>` so
`push_back` never invalidates pointers to live slots.
`ring.slot_state(id)` hands out a stable pointer for every live slot. Each slot
is reclaimed individually when it reaches CONSUMED. The deque is reset only
when the worker has no registered runs or live slots.

The HeapRing is **partitioned into `MAX_RING_DEPTH = 4` independent
rings** (Strict-1; matches L2's `CHIP_MAX_RING_DEPTH`). Each ring is its
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

When the child finishes the kernel, it writes `TASK_DONE` to the mailbox. The
Scheduler calls `LocalMailboxEndpoint::poll_progress`, which reads the mailbox
error fields and returns a `WorkerCompletion`. `MAILBOX_OFF_ERROR == 0` maps to
success; a non-zero child error maps to task failure. The endpoint lane reports
that completion through `Scheduler::worker_done`.

At this point:

- Host-backed output data has been copied back during native finalization
  and is visible through the shared backing. Device-backed output stays on
  its owning chip until the caller requests a copy
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
| 4 | L4 WorkerThread (PROCESS compatibility endpoint) | encode `(callable digest, config, args_blob)` into the base frame; write `TASK_READY`; wait for terminal state |
| 5 | L3 child `_child_worker_loop` | wake on `TASK_READY`; read digest → child-local slot → `my_l3_orch` |
| 6 | L3 child | `inner_worker.run(my_l3_orch, args, cfg)` → `scope_begin` → `my_l3_orch(orch3, ...)` |
| 7 | L3 `Orchestrator.submit_sub` | `l3_sub_handle` digest dispatched to L3's own sub worker child |
| 8 | L3 sub child | child resolves digest to its local Python callable and executes `verify_result()` |
| 9 | L3 run fence | all L3 tasks complete; `scope_end` + `wait_run` return |
| 10 | L3 child | `inner_worker.run()` returns; `_child_worker_loop` writes `TASK_DONE` |
| 11 | L4 LocalMailboxEndpoint | sees `TASK_DONE`; returns success completion |
| 12 | L4 run fence | L4 `scope_end` + `wait_run`; `w4.run()` returns |

Each level's orch fn receives **its own** `Orchestrator` — the recursion is
symmetric. `Worker` code does not branch on `level`; the level is only a
diagnostic label.

---

## 10. Worked example — one L3 chip task

Given a compiled `chip_callable` with signature `(IN, IN, OUT)` that accepts
three float32 tensors of shape `(N,)`:

```python
import torch
from simpler import Worker
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType

w3 = Worker(level=3, device_ids=[0], platform="a2a3sim",
            runtime="tensormap_and_ringbuffer")
chip_handle = w3.register(chip_callable)
w3.init()
try:
    a_buf = w3.create_buffer(N * 4)
    b_buf = w3.create_buffer(N * 4)
    c_buf = w3.create_buffer(N * 4)
    a = torch.frombuffer(a_buf.shm.buf, dtype=torch.float32, count=N)
    b = torch.frombuffer(b_buf.shm.buf, dtype=torch.float32, count=N)
    c = torch.frombuffer(c_buf.shm.buf, dtype=torch.float32, count=N)
    a.fill_(2)
    b.fill_(3)
    c.zero_()

    args = TaskArgs()
    args.add_tensor(a_buf.tensor((N,), DataType.FLOAT32), TensorArgType.INPUT)
    args.add_tensor(b_buf.tensor((N,), DataType.FLOAT32), TensorArgType.INPUT)
    args.add_tensor(c_buf.tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)

    def my_orch(orch, task_args, cfg):
        orch.submit_next_level(chip_handle, task_args, cfg, worker=0)

    w3.run(my_orch, args, CallConfig())
    # Read c here, while its backing is live.
finally:
    w3.close()
```

Step-by-step (one chip worker):

| Step | Where | What happens |
| ---- | ----- | ------------ |
| 1 | parent Python | user builds `args: TaskArgs`, calls `w3.run(my_orch, args, config)` |
| 2 | `Worker::run` | invoke the Python orchestration callback with its Orchestrator facade, args, and config |
| 3 | `Orchestrator::submit_next_level` | allocate a task slot; store its args; infer dependencies using canonical buffer identities and overlapping views; enqueue ready work |
| 4 | Scheduler thread | pop `slot` from worker 0's FIFO; resolve stable worker ID 0 to WT_chip_0; dispatch |
| 5 | WT_chip_0 parent side | encode one leased task frame: write `config`, digest prefix, and the args blob; publish `TASK_READY` for the active lane or `PREPARE_READY` for a staged successor |
| 6 | chip_0 child process | validate the frame and resolve its digest; ordinary HBG with an active predecessor also prepares the leased inactive arena bank before publishing `FRAME_STAGED`, while a frame with no active predecessor, diagnostic HBG, and TMR publish after validation and defer native prepare |
| 7 | chip_0 native-run path | after activation and the predecessor's finalization fence, launch an already-prepared HBG run or finish deferred native preparation and then launch; poll it to completion and finalize it before another staged frame may launch. Compatibility endpoints perform the equivalent operation through blocking `ChipWorker::run` |
| 8 | runtime.so | stage resolved host-backed tensors on the device; dispatch AICPU / AICore; copy output back to `c` during finalization |
| 9 | chip_0 child | native finalization returns; write `TASK_DONE` |
| 10 | WT_chip_0 parent | observe `TASK_DONE`; push success completion |
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
runtime: `ChipStorageTaskArgs` (`task_args.h`) is already declared with
`void` as the TensorTag parameter.

### Why no `WorkerPayload` wrapper

The chip child resolves the local slot and materializes the args before
calling the native ChipWorker with the chip POD and `CallConfig`. The
scheduler's parent-private `TaskSlot` is held by WorkerThread for
the completion callback and is not passed into the child. The distinct,
generation-safe `PipelineSlotLease.slot_id` does cross the mailbox boundary.

### Why slots on heap, mailbox on shm

Slots carry scheduler-only state (atomics, mutex, `std::vector` of fanout
consumers) that is parent-private. Putting them in shm would force cross-
process atomics and shm-safe containers. The only data that needs to cross
the fork boundary is per-task: callable, config, args — and that fits in a
fixed 64 KiB task frame with a one-time memcpy per dispatch. A two-frame-capable
local mailbox reserves a separate 64 KiB control base plus two such task frames;
single-frame compatibility endpoints use the base frame and leave the reserved
task frames unused.

### Why TaskArgs in slot (not encoded blob in slot)

`TaskArgs` is vector-backed. Storing an `uint8_t args_blob[N]` inline in the
slot would cap task size per level and waste memory per slot. Since the slot
is parent-heap, there is no fork-boundary constraint on what it holds — just
store the `TaskArgs` object and encode it into the mailbox blob at dispatch
time.

### Why `TaskArgsView` is just pointers + counts

`read_blob` constructs a non-owning view over the encoded `Tensor` and scalar
records. Counts allow bounds checks; byte storage plus the copying `tensors(i)`
accessor avoids assuming tensor alignment in the mailbox. Native chip execution
consumes materialized `ChipStorageTaskArgs`, not this view.

---

## Related

- [hierarchical-level-runtime.md](hierarchical-level-runtime.md) — L0–L6 level
  model, three-component composition
- [orchestrator.md](orchestrator.md) — how `submit_*` actually builds the DAG
- [scheduler.md](scheduler.md) — how dispatched slots get worker threads
- [worker-manager.md](worker-manager.md) — `WorkerThread`, mailbox
  layout, fork ordering
- [chip-level-arch.md](chip-level-arch.md) — L2 single-chip: three-program
  model (host / AICPU / AICore)
- [`../src/common/task_interface/task_args.h`](../src/common/task_interface/task_args.h)
  — `TaskArgsTpl` template and the `ChipStorageTaskArgs` alias
- [`../src/common/task_interface/task_args_wire.h`](../src/common/task_interface/task_args_wire.h)
  — the L3+ `TaskArgs`, `TaskArgsView`, and the mailbox blob codec
- [`../src/common/task_interface/tensor.h`](../src/common/task_interface/tensor.h)
  — `ChipTensor` POD and `TensorArgType` enum
