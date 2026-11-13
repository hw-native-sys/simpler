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

Supported Graph tasks use the resident scheduler. The host builds a separate
native scheduler image containing ENTER, child, and EXIT nodes, shared immutable
Definition plans, and per-instance argument storage. The public task image keeps
its original task IDs and layout. AICore control queues release roots at ENTER
and complete EXIT only after all sinks finish; control nodes consume no execution
slots. Invalid task shapes, noncanonical Definition argument spans, and
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
plans the resident ready pool from task metadata and launch topology. Output layout
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

The shared image uses three per-slot structures:

| Structure | Purpose |
| --------- | ------- |
| `TaskDescriptor` | Full task ID, kernel IDs, packed-buffer addresses |
| `TaskPayload` | Argument counts, predicate, dispatch metadata, and a delta naming each of its tensor, scalar and fanin regions — the arguments themselves live in the pool segments, so the payload is a fixed three cache lines regardless of the argument caps |
| `ChipTaskSlotState` | Active mask, attributes, block/subtask counters, completion state, task/payload bindings |

The host/device boundary is POD and position-independent. Fanins are integer
producer IDs, not pointers, and a slot state names its payload and descriptor — and
a payload names its three argument regions — by a delta from the naming field's own
address, so the image needs no fixup on either side of the copy. A delta is only
correct for the layout it was taken in, so `compact_live_image` re-takes every one
of them against the shipped image; the copy to the device moves a field and its
target together and leaves them all correct.

### 3.1 What Ships: Task Image and Resident State

A5's pooled runtime arena contains exactly the compact shared-memory task image.
`gm_sm_ptr` names the allocation's base, and `sm_image_bytes` is both its requested
size and its publication length. Every segment offset is relative to that base;
there is no runtime header or device-only prefix before it.

| Storage | Contents | Ownership and publication |
| ------- | -------- | ------------------------- |
| compact task image | shared-memory header, live task storage and argument/dependency pools | exclusive pipeline slot; one retained staging source and one image publication |
| resident scheduler backing | run control, worker contexts, ready queues, native graph image, dispatch payloads and optional async/diagnostic state | independent scheduler storage; published before the task image |
| graph heap | per-run task buffers and Graph execution argument/storage tails | independent pooled allocation; task addresses rebased before publication |
| Host orchestration context | `RuntimeContext`, `OrchContext`, `OrchestratorState` and borrowed scratch | Host only; never published |

A5 does not reserve, wire or initialize the common arena's `SharedMemoryHandle`,
`SchedulerState`, AICPU ready/early/sync/dummy/graph queues or completion mailbox.
Its launch descriptor's prebuilt-arena base and runtime offset stay zero. AICPU
starts from `scheduler_bootstrap` and the report/READY lifecycle; workers receive
resident contexts through that bootstrap, independently of the task image base.
A2A3 and WSE still use the common full arena lifecycle for their active executors.

The host-local `RuntimeContext` lives until the orchestration callback and its
recorders have joined. Its inline-completion scalar is bookkeeping for the common
orchestration API; the A5 scheduler plan derives completed nodes from the host
SM. A retained prepare plan stores callback results, leases and diagnostic sources,
so a resource-wait retry needs neither this stack object nor a second callback.
Commit attaches to the populated host mirror at its reservation pitch without
resetting its task count or execution header. This also holds across resource-wait
retries. The slot's staging remains alive until launch publication or failed-prepare cleanup.

`BindStaticArena` reports the compact image request. `BindImageStage` reports
`copied=0` and `bytes=sm=sm_image_bytes`; A5 emits no `BindArenaBuild` or
`BindRuntimeInit` phase. Resident ready-pool sizing and shape/capacity validation
apply to the actual execution plan, without an AICPU queue-population gate.

### 3.2 Bounded H2D Upload

The shared-memory mirror is dimensioned for the run's configured task count
(`runtime_env.ring_task_window`, default `CHIP_DEFAULT_GRAPH_TASKS`) but a run only
writes `[0, total_tasks)`, and the device boots scheduler-only and reads no SM slot
past `total_tasks`. So the SM H2D shipped each run is bounded, not capacity-sized —
the contract that keeps `bind` proportional to the workload.

The header is zeroed on the host; `descriptors`, `payloads`, `slot_states` and
`task_states` are each written per task at submit. Per-slot reset is
init-on-write in `orch::prepare_task` as each slot is claimed — there is no
table-wide reset. In the mirror those four live prefixes are a full reservation
apart, so `compact_live_image` restacks them (plus the three argument pools) into
an image pitched to `total_tasks`, where they are contiguous and travel as **one**
`copy_to_device`. The device attaches with the same pitch.

The mirror itself is the platform runner's, one buffer per pipeline slot, held
across binds and grown to the largest capacity any bind has asked for
(`HostApi::acquire_sm_mirror`). At the configured task capacity it is tens of MB,
so a per-bind buffer is an `mmap` and an `munmap` per bind. The block is handed
over uninitialized, so first touch still commits it and a run pays only for the
bytes it writes. Init-on-write is what makes the reuse safe, and reuse does not
weaken it: every byte a device-side reader reaches inside a shipped prefix is
written by the bind that ships it. The one shipped byte range no reader reaches is
the alignment padding the fanin cursor rounds past, which lies outside every
payload's `fanin_count`.

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

The public task payload stores fanins. For resident scheduling, Host converts
these through the unified graph accessor into a deduplicated fanout CSR and
initial remaining counters, excluding Host-completed edges, when selecting the CSR candidate. The immutable
header/CSR and atomic-only counters occupy separate 64-byte-aligned regions.

## 6. Boot Classification and Dependencies

The Host maker constructs deduplicated fanout CSR rows and remaining counters
by default. Resident completion traverses them synchronously. Valid images
without a dependency table use synchronous wake lists: bootstrap registers
each blocked node on its first unfinished producer. Invalid storage is rejected,
not routed through the other algorithm. Neither path exports a continuation.

A wake-list resolver atomically closes its producer head and owns the detached
chain. Each waiter's `wake_cursor` packs the next signed 32-bit task ref and the
next unexamined 32-bit fanin index into one atomic word. The resolver captures
both before routing can transfer the waiter. Registration publishes this word
before the head CAS. A successful CAS transfers ownership; the old owner never
reads or writes the waiter afterward. CLOSED producer heads are monotonic.
The separate `next_waiter` word belongs to control inboxes and Ready batches.

All completion owners resolve synchronously. `scheduler_resolve_wake_completion`
drains the detached chain, retaining each waiter's immutable fanin view and old
next link across Ready-prefix publication. `scheduler_route_detached_waiter`
transfers it to an unfinished producer or returns it for Ready publication.
The CSR candidate uses `scheduler_resolve_csr_completion` with the same ownership
contract. Complete resolver timing includes traversal, routing and publication.

Native control service handles at most eight nodes per invocation and fully
resolves each node. Async FINISH holds its exclusive lease until TaskDone and
all successor responsibilities are published, then terminates the lease and
clears its directory bit. Async condition CHECK retains its bounded polling;
FINISH has no GM progress record and LocalState has no resolver continuation.

## 7. Dispatch and Completion

Resident workers claim ready work, prepare their own slots, and call the selected
kernel. Ordinary tasks use bounded logical-block membership; MIX and sync-start
work share the cooperative admission order and explicit physical-lane ownership.
Each executed physical subtask contributes to the packed completion state. The
last physical contributor owns logical completion or the async join. Logical
TaskDone resolves all successors synchronously; cluster release only retires
resources and does not count as a business completion.

### 7.1 Resident Scheduler Local State

Each READY acquire initializes a fresh core-local configuration. Dispatch and
completion use cached worker IDs; the Executor derives payload addresses from
one shared region offset and the worker ID. Narrow offsets are validated before
conversion, and every cluster member must agree with the fixed payload stride.
All discovered clusters and all three lanes in each cluster participate.
Graph demand is checked against available capacity; it does not mask workers.
Partial-core masking has no current workload requirement and would add a
separate participation policy to bootstrap, dispatch and shutdown. All discovered
lanes therefore participate, and bootstrap partitions the task scan across their
Schedulers. This scan parallelism is not an isolated performance measurement.

The scan also validates and constructs each cooperative task's immutable
admission record and member bitmap. INITIALIZED grants no Ready ownership,
order key or block. The completed scan gate gives every Scheduler proof
that these records belong to the current run. Dependency routing remains dynamic:
the unique Ready owner retains the candidate in private pending storage and
publishes the record through the global commit lock. Members still check the run epoch,
committed order and ADMITTED signal; they reuse the scan's static membership
validation. No block counter can change before admission, and commit retries
retain the same pending owner. Task and completion counts are unchanged.

The validation contract distinguishes initialization from publication:

| Boundary | Checks |
| -------- | ------ |
| Bootstrap, before publishing INITIALIZED | Initial claim/next counters, completion progress and seal are zero; the packed completion total remains intact; record shape, flags and storage offsets match task metadata; member storage is empty; topology determines the frozen bitmap. |
| Each ready claim | Ready pop or direct resolve already grants unique ownership; a live run/error/epoch check and an empty private pending slot permit the local transfer. No shared phase or per-task lock is published. |
| Every commit attempt | A busy global lock returns RETRY after checking the active run. Under the acquired lock, active run/error, INITIALIZED, record epoch/order and log capacity are checked before publishing the record, ADMITTED and tail. Only its owner releases the lock. |
| Member adoption | The log task, run epoch, order key and ADMITTED publication are checked before the frozen bitmap is consumed. |

No legitimate writer changes a task's initial block counters before ADMITTED, or
its immutable record fields after the bootstrap scan. The fast path relies on
that protocol; it does not repeat a general corruption scan of those fields at
every commit or member adoption. Tests that inject invalid static data or initial
counters exercise rejection at bootstrap. Tests of retry-time epoch, order,
state, error and capacity violations exercise the production publication path.
Both ordinary and native resident entry points and low-level scheduler fixtures
bootstrap their complete task configuration before dispatch. There is one
preparation lifecycle: UNCLAIMED to INITIALIZED at bootstrap, then ADMITTED on
commit. No construct-on-claim fallback is retained.

Task-level Mix membership is the active lane mask plus the AIC coordinator.
A 1C1V task therefore coordinates two lanes per cluster; 1C2V and 2V coordinate
all three. The inactive AIV has no obligation for a two-lane group. Per-lane
assignment and predecessor sequences preserve descriptor ownership when the
resource mask changes between groups. This task-level selection does not remove
workers from bootstrap or shutdown.

Ordinary single-type SPMD tasks admit at most one eligible worker per block.
Their frozen membership is a cyclic interval in ascending worker-index order,
starting at `member_rank_offset % eligible_count` when the block count is
smaller than the eligible set. Host computes the prefix of ordinary non-sync
SPMD blocks separately for AIC and AIV, in task-ref order, and stores it in
the existing 64-byte admission header. Other resource types, single-block
work, MIX and sync-start do not advance that prefix. This distributes independent small tasks across
workers without changing block ownership or completion accounting. MIX and
`sync_start` retain their full eligible membership. The ordinary and native Graph tags remain v10 and v11.
Both carry the current subtask-completion fields, per-resource-type member
prefix and cooperative admission. Historical field layouts are rejected.
Each dispatch subtask contributes once; MIX resource retirement does not
contribute again. Execution identities use the strict v1 contract:
`(run_epoch, task_ref, block_idx, subtask_slot)` identifies an execution;
worker and pending slot identify its placement. Worker event sequences prove
that retirement precedes the next payload publication and permit on a reused slot.

A completion resolver may retain one dependency-ready cooperative successor in a
persistent admission candidate, independent of the completing core's type. It
attempts the ordinary claim/prepare/commit sequence before returning; a busy
commit lock retains pending ownership for the next finite scheduler pass.
A full candidate set sends successors through the shared queues. This does not
reserve an execution slot, bypass any fanin, or grant a kernel entry permit.
Ordinary direct refill remains limited to the executing core type and excludes
MIX. Queued tasks preserve their queue's FIFO order; already-ready independent
tasks do not impose an ordering on this direct admission.

Kernel Ready work uses exactly two shared ticket queues: AIC and MIX tasks
enter the AIC queue; ordinary AIV tasks enter the AIV queue. Each worker probes
only its own core type. The host sizes capacity from logical task counts and
publishes a version-11 header. Queue head, tail and entries use atomic-only
publication words; entries remain immutable until the run fence. Native control
nodes retain their separate control inboxes.

A ticket reservation and publication complete in one call. Synchronous
FINISH/control resolution uses a private eight-ref publication buffer;
no ready-pool publication is held in a worker-local pending list. Historical
alternative algorithms and the requirements for implementing them again are
recorded in the [queue design](../../../../../docs/zh-cn/design/a5-hbg-graph-task/ready-queue-design.md).

Business-ready ticket positions are shared GM state; a successful head CAS
transfers task ownership to one worker. Native control inboxes retain their
separate atomic endpoints and task links.
Run entry validates the immutable graph, worker topology and queue layout.
Internal helpers require those live bindings instead of repeating pointer and
configuration checks. Shared queue references and resumed cursor state are
validated when consumed; dynamic head, count, epoch and ownership checks remain.
Ticket claiming uses the shared queue directly and does not take a scheduler
count parameter.

Candidate classification reads the immutable metadata flags. Host construction
and record validation establish that SPMD means more than one logical block;
MIX and sync-start are encoded alongside it. The ordered discovery and ordinary
preparation passes therefore do not load the complete metadata just to classify
a candidate. Payload preparation still reads its kernel IDs and execution shape.
Two own slots use a ready mask and head index to preserve publication order.
The executor peeks at the head and checks final admission before consuming it
and changing READY to EXECUTING. A denied permit leaves the order intact;
parking removes a candidate's slot, and preparing it again appends it at the tail.
Kernel return, async sealing and enabled argument capture precede the loop's
EXIT/error gate and a direct `scheduler_complete_own_slot` call. Retirement
retains the executor's private slot ownership through normal kernel return:
the kernel receives the GM argument vector and payload contexts, not private
slot/admission addresses. The checked completion entry validates bindings for
other callers. There is no completion notification or periodic completion scan.
Direct refill preserves the old execution in a local snapshot before rebinding
the slot to its successor. The executor carries only its validated async group
index across the kernel call and rebuilds the writer from its immutable run
layout; mutable registration state is still validated when sealing the group.

Local configuration, execution slots, control resolution and the optional
profiling pointer belong to the core-local state. A separate profiling state
holds two timing slots, self-execution traces, one worker trace cache, profiling
offsets, and the loop counter/cache-valid flag. Slot retirement precedes physical
completion contribution. Completion trace publication precedes GM MIX lane release.
Sampling is derived from the timing-slot range.
Retirement reads the executor's trace in place: the same core owns that storage,
and it cannot be rebound until the current retirement has finished publishing.

The host records whether any task requests sampled timing in the run control.
Only a run with chip profiling or sampled timing enters the resident function
specialization that allocates profiling state; the plain specialization does
not allocate it. Four `resident_*.cpp` translation units instantiate the
ordinary/native and profiled/plain combinations from `resident_scheduler_impl.h`;
`CoreType` distinguishes AIC and AIV entry symbols. The shared completion resolver
is a TU-local `static noinline` function with natural-alignment guards on its
local argument types. Struct
sizes exclude other function locals, worker statistics and compiler spills,
and do not establish the dynamic AICore stack high-water mark.

The Scalar policy is local to each translation unit. Policy-dependent memory
helpers, their scheduler callers and observer-bound adapters have internal
linkage, so emitted inline bodies cannot cross from plain to profiled residents.
Shared wire types and the exported resident entry points retain their common
identity. AICore plain residents disable instrumentation at compilation; CPU
simulation residents retain observation. The Scalar trace test links both
policies together and exercises emitted memory, ordered-poll and async bodies.

Before bootstrap, every participating core invalidates its entire data cache.
The callable table and task metadata are immutable throughout that run, allowing
callable lookup and completion resolution to omit repeated invalidation of those
immutable lines.

Each worker owns two GM DispatchPayload slots and their core-local bookkeeping.
Ordinary, SPMD and GM MIX execution reuse these same slots. The worker prepares
its own payload, publishes local readiness, executes it, and directly retires it after execution. The scheduler does not reserve or use cluster SSBUF.

Each slot's embedded context pointers are initialized in the host-created
payload image. Dispatch updates `block_idx` and `block_num` for the current
logical block and task. Self-execution consumes its own
payload writes after the publication barrier, without an additional payload
invalidation on pickup. Each invocation initializes fresh local state before
bootstrap; AICPU waits for `bootstrap_complete` before releasing the
DATA_MAIN_BASE launch gate.

For ordinary CSR graphs, the Host seeds ready tickets from the initial fanin
counts and writes the immutable cooperative admission members before uploading
the scheduler image. AICPU validates the compact launch topology before READY.
The image's scan-complete marker lets each worker proceed directly to local
slot preparation. Admission ownership and block claims still occur after
dependency resolution. For even-sized partial AIV waves in a graph with MIX
work, the Host rounds the starting type rank to a cluster boundary to keep
whole clusters available for other work. Native Graph and wake-list graphs
use the device scan.

Resident self-execution retires its EXECUTING slot directly on kernel return,
after sealing asynchronous condition registrations and checking the run boundary. Computation contribution
and successor resolution precede trace consolidation and writeback. Every trace
is published before local slot reuse, MIX lane release and EXIT acknowledgement;
the Host reads these records only after all cores have acknowledged retirement.

The dispatch timing interval ends at `ready_publish_cycles`, immediately before
READY token publication. It excludes READY publication and all subsequent
dispatch-trace GM writes. Device RunWall measures the full on-NPU run
independently of this phase boundary.

Ordinary resident runs accumulate resolved counts locally until an idle pass
with no local executable task or deferred reservation, or the common run epilogue.
Native Graph runs also flush pending counts when 32 completions have accumulated
or at least 1 ms has elapsed since the previous flush. The flushing core's trace
and error publication precedes its count flush; other contributors finish their
trace publication before their own EXIT acknowledgement. The AICPU watchdog remains 20 seconds by default, so
continuously busy ordinary work without count publication can reach that timeout.
The intended ordinary runs normally complete in under one second. Ordinary
completion accounting avoids periodic publication on the busy path; its
continuous-busy timeout is an accepted limit outside that expected duration,
not evidence that busy execution has stopped making progress.

MIX tasks use ordered admission and GM group mailboxes. The AIC coordinator
assigns a logical block to the participating lanes. Each required resource lane
publishes arrival after its local slots are free. After all required arrivals,
lanes with subtasks prepare and execute their own slots, then publish release
after slot retirement. Resource lanes without a subtask release without preparing
a slot.
The coordinator joins the required releases and retires the block; the final
block closes computation and resolves dependencies, or enters async completion
when required. Task traces retain the logical task/block/subtask identity.

Ordinary dispatch skips ready-pool queries when no usable local slot can accept
work. Ordered discovery is bounded by candidate credits instead: it can claim
and admit a cooperative task while all execution slots are occupied.
Idle polling backs off from 8 to at most 128 iterations. Dispatch trace GM
writes follow ready publication; Host tooling computes ready-to-kernel latency
from the recorded timestamps.

### Preparation does not reserve future execution

Bootstrap preparation initializes immutable membership and validation data only.
It does not claim ready ownership, append an admission order key, or reserve a
worker obligation or execution slot. A successor becomes eligible for admission
only after dependency resolution observes its predecessor's logical completion,
including all registered asynchronous completion conditions.

A sole DAG predecessor is not a proof of progress independence: that predecessor
may wait on a COUNTER published by an otherwise independent task. Admitting its
successor before logical completion can occupy the workers required by that
publisher and create a resource cycle. Completion-time direct handoff avoids a
ready-queue round trip only after the normal completion and fanin checks pass.

## 8. Scalar Access During Construction

`get_tensor_data` and `set_tensor_data` operate only on explicit `HOST/NONE`
arguments with registered bounds and compatible directions. Device arguments
and graph-heap allocations are refused; there is no implicit mapping or device
copy fallback. See [SCALAR_DATA_ACCESS.md](SCALAR_DATA_ACCESS.md) and the shared
[HBG host data access contract](../../../../../docs/hbg-host-data-access.md).

## 9. Errors and Diagnostics

The runtime latches a fatal code and maps it to the negative run status the host
observes. The two reporters latch in different places: a scheduler code goes into
the shared-memory header, and an orchestration code into
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

Pure scheduler/core-tracker and lifecycle primitives also have C++ unit tests
under `tests/ut/cpp`.
