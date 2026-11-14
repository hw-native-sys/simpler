# `host_build_graph` Runtime Design

## 1. Execution Model

`host_build_graph` separates graph construction from device execution:

```text
host register: materialize + dlopen orchestration SO
        ↓
host run/bind: stage external tensors, execute orchestration to completion
        ↓
host: copy the prebuilt graph image to device memory
        ↓
device: validate the launch, bootstrap resident scheduling, execute tasks
        ↓
host: collect outputs and destroy/reset per-run state
```

The device has no orchestration thread. The resident scheduler runs on the discovered
AIC/AIV workers; AICPU initializes, monitors, and tears down the workers.
The AICPU build contains no AICPU dispatch state or GraphExecution materializer.
Percounter selects the common materializer separately for its Graph preparation.

Supported Graph tasks use the resident scheduler. The host builds a separate
native scheduler image containing ENTER, child, and EXIT nodes, shared immutable
Definition plans, and per-instance argument storage. Host binding prepares Calls before launch;
execution tokens identify block/lane work on the device. The task image preserves
original task IDs in an A5-owned compact layout. Bounded AICore scans complete
ENTER after its producers are DONE; child roots observe ENTER's canonical gate.
EXIT completes only after every sink is DONE. Control nodes consume no execution
slots or ready inbox. Invalid task shapes, noncanonical Definition argument spans, and
insufficient argument capacity fail preparation. The native image and binding contracts are specified in
[the Graph task design](../../../../../docs/zh-cn/design/a5-hbg-graph-task.md).

This ordering is the defining constraint of the runtime. The host constructs the
whole graph before any device task can complete.

## 2. Callable and Run Lifecycle

### 2.1 Registration

`register_callable_impl` materializes the orchestration bytes as a temporary
shared object, opens it with `dlopen(RTLD_LOCAL)`, and resolves:

- `aicpu_orchestration_config`;
- `aicpu_orchestration_entry`; and
- `framework_bind_orch_context`.

The resolved handle and entry points belong to the registered callable and stay
alive across prepared runs. Unregister/context teardown closes the handle and
removes its temporary file.

Child AIC/AIV callables are uploaded separately. Their resolved device function
addresses populate the per-run dispatch table.

### 2.2 Host Graph Construction

For each run, the host:

1. validates and stages external tensors into a run-owned host accessor;
2. acquires a retained host shared-memory mirror and initializes a host-local orchestration context;
3. binds the runtime to the orchestration DSO;
4. calls the orchestration entry synchronously;
5. finalizes task counts and records the prepared metadata sources in Runtime's host-only pending publication;
6. returns from bind, then publishes resident scheduler metadata and the compact
   shared-memory image through launch-owned staging.

The exclusive pipeline slot retains only `OrchestrationScratch`: TensorMap
arrays and the fan-in epoch array. Each bind owns a fresh `OrchestratorState`,
borrows those arrays, and returns them after Graph recorder teardown on every
exit. Fatal state, counters, mirror/heap pointers and Graph bindings belong to
that run. A bind also holds a local owning reference. `ChipWorker::finalize()`
drains native runs, finalizes and destroys the runner, and only then closes
`host_runtime.so`, which owns the scratch deleter and shared-pointer control-block
code. A failed device finalization retains both the runner and module for retry.
Callable and executor libraries closed inside runner finalization are separate
modules. `HostApiOps` is host-only: all providers and consumers must use the same
table definition and be built together. Null fallback covers omitted callbacks
in that definition, not shorter tables from older binaries.

TensorMap resets bucket heads, every retained task head and allocation cursors without
reconstructing unused entries. Bucket and entry-pool storage is reused while
its own size matches; task-head and fan-in storage grow only. Changing the
active task capacity does not reconstruct the unchanged entry pool. A failed
TensorMap resize preserves the prior layout and contents; a successful init
empties the map. With default capacities the arrays retain approximately
8.72 MiB per used slot until finalization. Pool-size changes can temporarily
hold old and replacement arrays. No new lock or device allocation is involved.

The shared orchestrator accumulates queue-population counters when a task slot
receives its final classification. A5 does not aggregate Graph definition
populations or use those counters to size resident queues. `create_scheduler_state`
plans immutable engine catalogs and prepared executions from task metadata and launch topology. Output layout
entries are initialized and read only at `OUTPUT` argument positions.

`publish_run_image_impl` is the single metadata consumer. Definition and compact
image sources are retained by the exclusive pipeline slot; A5 scheduler source
storage moves into the same pending record. Device allocations keep their
existing owners. No rebind/growth may replace borrowed staging before the copy
returns. Consuming the record also consumes it on failure: later regions are not
written, launch is forbidden, and failed-prepare cleanup retires run-owned
allocations. Abandoning a bind clears its sources without performing metadata
copies. Tensor copy-in and host get/set remain preparation effects; failed
publication cannot undo them.

One owner-checked host phase trace covers bind and publication. GraphUpload and
ArenaH2d measure their actual successful copies and preserve byte-count details;
packing is preparation work. A completed run's release cannot close a successor's
trace. Publication is synchronous; these lifetimes do not cover asynchronous DMA
or captured graphs.

An orchestration fatal stops this sequence before the upload. The orchestrator
runs on the host, so its code is latched in `OrchestratorState::fatal_code` and
never reaches shared memory; the bind maps it onto the status the caller sees.

Step 1 gives each caller tensor a slice of the runner's retained temporary buffer
rather than a per-run `device_malloc` / `device_free` pair: bind packs the run's
HOST/H2D tensors to a 1024-aligned required size, grows the buffer only when a
run needs more than is currently retained, and bump-slices each tensor from it,
so a steady-state workload performs no temporary device allocation at all. The
slices are recorded as `BufferNoop` leases — validate copies the written ones
back and releases none, and the buffer itself is freed once at Worker finalize.
The buffer is per pipeline slot; a run holds its slot from bind through validate
and a concurrent reservation is admitted only on a distinct slot, so no other
run can re-slice a buffer whose slices are still live. The mechanism
(`RetainedTempBump`, `src/common/utils/retained_temp_bump.h`) is shared with
`tensormap_and_ringbuffer`; that runtime's `RUNTIME_LOGIC.md` §2.4 carries the
grow/slice details.

HOST/NONE arguments bypass the temporary buffer. They remain HOST views for
host orchestration's `get_tensor_data` (IN/INOUT) and `set_tensor_data`
(OUT/INOUT). HOST/H2D arguments become DEVICE views after binding and cannot be
accessed through those host entries. DEVICE task and Graph boundary arguments
must be DEVICE. When both sides need the same control data, the caller passes
separate HOST/NONE and HOST/H2D (or DEVICE/NONE) arguments; host writes do not
mirror into the device argument. Device-produced host controls require an
explicit completed copy before submitting the dependent call.

An empty caller tensor addresses nothing, so it takes no slice and reaches
orchestration with a null address. **This is a change in what hbg accepts.**
Before the retained temporary buffer, a zero-byte non-child tensor that was not a pure
`OUT` failed the bind: it was handed to `HostTensorAccessor::add`, which rejects
an empty region, and the bind reported `no host view for tensor N`. It is now
passed through, which is what `tensormap_and_ringbuffer` has always done.

A pure `OUT` tensor gets a slice but is never copied in, and nothing
zero-fills it, so the bytes a kernel does not write are whatever the slice last
held. That was already true of a fresh `device_malloc` — the allocator pools and
reuses device memory — but the residue is no longer always arbitrary. **On a run
that reuses the retained buffer unchanged, it is whatever the previous tensor
occupying that byte range left there.** The slot is keyed by pipeline slot
alone, not by callable or by tensor identity, so that is the same tensor's own
bytes only when the same callable runs consecutively with the same argument
layout; a slot alternating between callables leaves another callable's bytes.
A run that allocated or grew the buffer gets uninitialized allocator memory as
before, and so does the first run of any Worker, because
`RetainedTempBump::begin()` neither preserves nor initializes a buffer it
replaces. So a kernel that writes only part of its output fails
deterministically across the steady-state rounds of a fixed-shape workload —
where a golden check is least likely to catch it — and randomly everywhere
else. `device_memset` exists for zero-filling pure outputs;
neither runtime uses it, because it would cost a device operation per output per
run.

### 2.3 Device Execution and Teardown

Every AICore reports its physical identity and the current launch epoch before
waiting for the AICPU. AICPU threads discover disjoint worker partitions; the
leader checks topology and configures the per-worker contexts. GM READY publishes
the contexts. Workers bootstrap their graph partitions while the execution gate
remains closed; AICPU opens the DATA_MAIN_BASE gate only after bootstrap succeeds.

A5 HBG accepts only resident modes 1/2. An invalid mode, null or misaligned
context base, or overflowing worker-context address sends an AICore into a
failure wait that never dereferences scheduler backing. It still reports its
identity and participates in EXIT, nonce ACK and teardown release.

AICPU validates the selected scheduler after discovery. A rejected launch latches
`INVALID_ARGS` and retires the discovered cores through the same partitioned
EXIT/ACK protocol as other failed initialization. The final failed-init owner
publishes a failed terminal record with the retirement result; an unproved
retirement cannot license storage reuse. Each active runtime extension validates
its own mode and context through its scheduler steps.

Discovery assigns each physical register window one retirement owner. Duplicate
or missing reports remain unproved. Initialization failure does not release a
phase join before its owner finishes configuring shared state. The leader publishes
the final initialization outcome once; a partition already released when startup
fails still signals EXIT and retires its workers. An invalid AICPU participant
count uses callback zero's platform startup-abort owner rather than entering an
executor rendezvous with an invalid participant count.

Normal shutdown starts only after graph completion or a recorded failure. Every
AICPU partition signals EXIT, collects acknowledgements, closes and drains its
register windows, and releases its teardown gates. The last participant settles
diagnostics only when retirement is proved, selects the terminal result, and
publishes cleanup eligibility. Exactly one cleanup owner deinitializes the
executor and hands off that result. TaskDone, core retirement and terminal
publication are separate events. PMU settlement additionally requires this run
to have initialized its counters; failed startup cannot restore a previous run's
collector snapshot.

## 3. Prebuilt Graph Image

Host orchestration uses the common `ChipTaskStorage` mirror as scratch. The
published default A5 HBG image has a separate POD representation:

| Structure | Bytes | Purpose |
| --------- | ----- | ------- |
| `TaskImageStatus` | 64 | First-error atomic at offset zero; reset by packing each run |
| `TaskImageStorage` | 80/task | 24-byte ID/kernel descriptor and 48-byte argument/predicate payload, padded to 16-byte alignment |
| fanin pool | 4/element | Original top-level producer IDs |
| tensor pool | 128/element | Tensor descriptors with committed device buffer addresses |
| scalar pool | 8/element | Scalar bits, never treated as heap addresses |

The image has no `ChipTaskSlotState`, task-state array, wait/early-dispatch state,
packed-buffer bounds or per-task dump metadata. Host planning obtains those from
the scratch mirror before launch. Live progress belongs to resident control
records. Dump selection publishes its own bounded argument-region descriptors.

`task_image::pack` projects each submitted record and copies only the live pool
prefixes. Each argument reference is a signed 32-bit delta from the naming field
itself. Packing binds the deltas against the destination image and rebases only
tensor-buffer and predicate addresses in the virtual heap window. Moving the
whole image to GM preserves every delta. Every shipped record and padding byte
is initialized during packing; the full image must fit within `INT32_MAX`.

Percounter owns a separate packer and layout through its runtime include paths.
Its current AICPU Graph materializer requires the common task representation.
Each runtime compiles one format; default HBG has no alternate-format reader.
A2A3/WSE continue using their common image and active executors.

### 3.1 What Ships: Task Image and Resident State

A5's pooled runtime arena contains exactly the compact shared-memory task image.
`gm_sm_ptr` names the allocation's base, and `sm_image_bytes` is both its requested
size and its publication length. Every segment offset is relative to that base;
there is no runtime header or device-only prefix before it.

| Storage | Contents | Ownership and publication |
| ------- | -------- | ------------------------- |
| compact task image | 64-byte error status, 80-byte task records and live argument/dependency pools | exclusive pipeline slot; one retained staging source and one image publication |
| resident scheduler backing | run control, worker contexts, engine catalogs, native graph image, prepared Calls/bindings and optional async/diagnostic state | independent scheduler storage; published before the task image |
| graph heap | per-run task buffers and Graph execution argument/storage tails | independent pooled allocation; task addresses rebased before publication |
| Host orchestration context | `OrchContext`, `OrchestratorState` and borrowed scratch | Host only; never published |

A5 does not reserve, wire or initialize the common arena's `SharedMemoryHandle`,
`SchedulerState`, AICPU ready/early/sync/dummy/graph queues or completion mailbox.
Its launch descriptor's prebuilt-arena base and runtime offset stay zero. AICPU
starts from `scheduler_bootstrap` and the report/READY lifecycle; workers receive
resident contexts through that bootstrap, independently of the task image base.
A2A3 and WSE still use the common full arena lifecycle for their active executors.

A5 orchestration needs no RuntimeContext. Its plan derives completed nodes from
the host SM. A retained prepare plan stores callback results, leases and diagnostic
sources, so a resource-wait retry needs no second callback. The full-arena
initializer and mailbox are built only for A2A3/WSE; their makers explicitly copy
the orchestrator's inline-completion count into their device RuntimeContext.
Commit attaches to the populated host mirror at its reservation pitch without
resetting its task count or execution header. This also holds across resource-wait
retries. The slot's staging remains alive until launch publication or failed-prepare cleanup.

`BindStaticArena` reports the compact image request. `BindImageStage` reports
`copied=0` and `bytes=sm=sm_image_bytes`; A5 emits no `BindArenaBuild` or
`BindRuntimeInit` phase. Resident catalog/Call sizing and shape/capacity validation
apply to the actual execution plan, without an AICPU queue-population gate.

### 3.2 Bounded H2D Upload

The shared-memory mirror is dimensioned for the run's configured task count
(`runtime_env.ring_task_window`, default `CHIP_DEFAULT_GRAPH_TASKS`) but a run only
writes `[0, total_tasks)`, and the device boots scheduler-only and reads no SM slot
past `total_tasks`. So the SM H2D shipped each run is bounded, not capacity-sized —
the contract that keeps `bind` proportional to the workload.

Host submit initializes each claimed scratch slot. Packing writes a fresh error
status, projects `[0, total_tasks)` into compact records, and restacks live pools
at cache-line-aligned boundaries. There is no device attach/reset of Host slot
state. Publication retains one staging source and performs one image copy.

The mirror itself is the platform runner's, one buffer per pipeline slot, held
across binds and grown to the largest capacity any bind has asked for
(`HostApi::acquire_sm_mirror`). At the configured task capacity it is tens of MB,
so a per-bind buffer is an `mmap` and an `munmap` per bind. The block is handed
over uninitialized, so first touch still commits it and a run pays only for the
bytes it writes. Init-on-write is what makes the reuse safe, and reuse does not
weaken it: every byte a device-side reader reaches inside a shipped prefix is
written by the bind that ships it. Alignment padding lies outside each payload's argument counts.

## 4. Whole-Graph Capacity

The runtime uses one task table, one graph heap, and one TensorMap pool. They are
capacity-bounded storage, not streaming flow-control buffers:

- the task table and the graph heap are forward-only bump allocators;
- task slots and heap bytes are never recycled mid-run; and
- TensorMap entries are held for the whole run.

There is no reclaim channel from the scheduler back to the allocator, so the
allocators carry no reclaim pointer and no back-pressure wait. A task id is
therefore also its slot index: ids run `0..capacity-1`, never wrap, and every
segment is indexed by the id directly — there is no slot mask, so the capacity need
not be a power of two.

Resident completion is published in the node control state and resolved-run
counters; it reclaims neither task slots nor heap. Host-completed nodes seed
bootstrap completion separately.

There is no post-run sweep that makes graph space reusable. Runtime destruction
releases the complete arena, and the next run starts from a newly initialized
image.

### 4.1 Allocation Failure

The graph must fit the configured task count, the fanin capacity, and the TensorMap
pool. The task count comes from `runtime_env.ring_task_window` (default
`CHIP_DEFAULT_GRAPH_TASKS`); the host mirror is allocated at that size and
committed by first touch, so a run pays only for the slots it writes. Because
nothing is reclaimed, a request that does not fit can never become satisfiable —
the allocator names the exhausted resource and fails on the spot. There is no wait
and no timeout.

Representative allocator output is:

```text
FATAL: Graph Too Large!
The whole graph must fit at once; nothing is reclaimed mid-run.
  Tasks:      used=.../...
  Graph heap: used=.../..., available=...
  Requested:  ... bytes + 1 task slot
```

This is host-orchestration logging. The allocator records the corresponding
runtime error and unwinds; it does not terminate the process directly.

The graph heap is not one of those capacities. Orchestration allocates it out of
a virtual window, and its device region is committed afterwards at the size the
graph turned out to need, so a graph too large for the device fails at that
commit — which names the byte count it asked for — rather than in the allocator.

## 5. Submission and Dependencies

### 5.1 Mixed Tasks and Logical Blocks

An active mask selects AIC, AIV0, and AIV1 lanes. `block_num` is a logical SPMD
width and may exceed the physical device width when sync-start is not requested;
the scheduler dispatches those blocks in waves.

Before a slot is allocated or published, submission requires:

```text
block_num >= 1
block_num * popcount(active_mask) <= INT16_MAX
```

The product is stored in the 16-bit `total_required_subtasks` field. Sync-start
adds a separate co-residency limit: AIV tasks use the available AIV count, while
AIC/MIX tasks use the available cluster count.

### 5.2 TensorMap and Fanins

TensorMap maps tensor regions to producer task IDs. For every task:

1. INPUT/INOUT regions look up overlapping producers.
2. Explicit and discovered producers are deduplicated into the payload's fanin
   region.
3. OUTPUT/INOUT regions register the new task as producer.
4. Each producer tracks its highest consumer local ID for completion metadata.

The compact task payload stores original top-level fanins. Native Graph access
maps them through `completion_refs`; child fanins use Definition-local indices,
and roots depend on ENTER. EXIT depends on every sink. The canonical producer
completion gate is the readiness proof for executable and control consumers.

## 6. Bootstrap and Dependencies

After READY acquire, each worker validates its context, graph view, epoch,
prepared layout and immutable engine catalog. Workers partition node ranges and
prepare required cooperative admission records. Cluster coordinators clear their
own signal before the bootstrap scan barrier. No Graph child argument binding
runs on the device: Host has already published those arguments and Calls.

After `bootstrap_scan_complete`, workers initialize two FREE local slots,
service bounded control-node scans and prepare their first executions. The last
worker publishes `bootstrap_complete`; AICPU then opens the DMB launch gate.
No kernel may enter while this gate remains closed.

ENTER, EXIT and inline control nodes use bounded scans rather than a ready inbox.
A worker reads each producer's canonical gate and advances only a positive fanin
prefix. When every producer is DONE, a unique CAS changes the control node's
completion gate from its initial state to its completed state. Only the winner
adds a pending logical completion. There is no control-side fanout/remaining or
wake-list notification path in this protocol.

## 7. Discovery, Dispatch and Completion

### 7.1 Prepared Calls and Execution Tokens

The Host builds a 16-byte `SchedulerExecutionPlan` for every node and one
512-byte immutable `SchedulerPreparedCall` for each physical execution. A Call
names its function, epoch, node, block, lane and complete argument vector.
Tensor arguments are GM descriptor addresses; scalar arguments are value bits.
`args[48]` and `args[49]` name the execution's 128-byte binding, containing its
LocalContext and GlobalContext on separate cache lines. Calls and bindings have
execution lifetime, independently of the worker's two private dispatch slots.

Each execution has a 128-byte atomic-control sector. Phase, owner and transfer
generation share one atomic state word. Fresh Calls start BUILT. A successful
block ticket and BUILT-to-CLAIMED CAS grant one worker the execution token;
Call acquisition observes and validates the immutable Call and binding. The
claim alone grants no permission to consume business input data.

Each worker rotates through its immutable engine catalog. The two catalogs
classify AIC/MIX and AIV tasks; neither has a dynamic enqueue, shared head or
FIFO contract between independent tasks. Cooperative discovery and obligations
share the loop with ordinary token discovery.

A core has four execution-token positions and two 40-byte dispatch slots.
A token may remain WAIT_FANIN while caching a confirmed positive producer prefix.
It owns a concrete execution but occupies no business slot. When the token
window is full, discovery can find a runnable replacement and publish a blocked
token through PARKING to PARKED before releasing its local position. Other
workers recover PARKED executions through bounded probes and periodic sweeps.
The parked-count hint cannot suppress the authoritative sweep. Owner and
generation changes always use atomic compare/exchange.

After fanin completes, a token transfers into a FREE slot as READY. Graph
argument descriptors are observed, and predicates are evaluated under their
supported single-block/lane shape. A false predicate skips the function while
retaining normal execution-return obligations. A direct execution path or the
local ready mask/head selects the next slot. The final shared admission,
cluster/MIX and epoch checks remain required before READY becomes EXECUTING.

### 7.2 Business Execution and Cooperative Work

The executor calls the prepared function with `Call.args`. The kernel receives
GM arguments and contexts, never a private slot or admission-state address.
Before/after args-dump boundaries coordinate with AICPU's bounded dump service.
Async input observation occurs on the actual consumer after dependencies settle.

MIX and sync-start work use the ordered cooperative admission records and
explicit physical membership. Group reservations bind each participant to its
own prepared execution. Reservation, preparation, arm and final permission are
separate ownership boundaries. A busy worker retains its existing obligation;
background discovery cannot displace an admitted participant or manufacture a
release. Cluster signaling and cache publication remain part of group ownership.

After the kernel returns, its async registration is sealed. The execution's
control moves from CLAIMED to RETURNED under its owner/generation proof. The
real physical return contributes once to the task's canonical completion gate;
inactive MIX resource lanes contribute no execution. Trace/group retirement
finishes under its existing ownership before the slot is safely reused.

### 7.3 Canonical Completion and Retirement

`completion_state` encodes immutable total in bits 63:48, remaining computation
in bits 47:32 and debt in bits 31:0. Initial state for T physical executions is
`(T,T,T)`. A real return atomically subtracts `(1 << 32) + 1`, reducing both
remaining computation and compute debt. Async conditions retain additional debt
until their join completes. DONE requires a valid total and zero computation
and debt; a kernel return alone is insufficient.

The unique final completion owner adds one pending logical completion. Workers
flush pending counts to `resolved_task_count`; control nodes use the same
canonical gate and winner rule. Successors observe producer gates directly.
Async CHECK is bounded and FINISH retains its exclusive lease until its debt
settlement completes. Trace publication, group release and safe reuse still
have their own ownership requirements.

On an error, a worker cancels only its provably owned unstarted handles to FAILED;
it cannot invent a RETURNED Call, computation contribution or group-finished ACK.
PARKED/shared armed work is revoked by the failed run's all-core retirement fence
before a later epoch initializes reused storage. AICPU sends EXIT, requires the
current ACK nonce, closes/readbacks register windows, and publishes
`post_close_release`. Only then may worker entry points return. Logical DONE or
a missing report is not proof of physical retirement.

## 8. Scalar Access During Construction

`get_tensor_data` and `set_tensor_data` operate only on explicit `HOST/NONE`
arguments with registered bounds and compatible directions. Device arguments
and graph-heap allocations are refused; there is no implicit mapping or device
copy fallback. See [SCALAR_DATA_ACCESS.md](SCALAR_DATA_ACCESS.md) and the shared
[HBG host data access contract](../../../../../docs/hbg-host-data-access.md).

## 9. Errors and Diagnostics

The runtime latches a fatal code and maps it to the negative run status the host
observes. The two reporters latch in different places: a scheduler code goes into
the compact image's `TaskImageStatus`, and an orchestration code into
`OrchestratorState::fatal_code` in host memory, since this runtime's orchestrator
runs on the host. Important validation paths include:

- invalid arguments (`-5`);
- sync-start residency violations (`-7`); and
- scheduler timeout (`-100`).

Device logs contain scheduler records only. Host graph-construction diagnostics
remain host-side. See [device_log_profiling.md](device_log_profiling.md).

## 10. Verification

Runtime C++ changes require rebuilding the editable package, then running both
simulation variants:

```bash
pip install --no-build-isolation -e .
pytest examples tests/st --platform a2a3sim --runtime host_build_graph
pytest examples tests/st --platform a5sim --runtime host_build_graph
```

Resident scheduler and lifecycle primitives also have C++ unit tests
under `tests/ut/cpp`.
