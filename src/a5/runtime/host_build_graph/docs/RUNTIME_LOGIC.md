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
- `aicpu_orchestration_entry`;
- `framework_bind_orch_context`; and
- `framework_finish_orch`.

Both bind and finish callbacks are mandatory. Finish releases the per-bind registry
before the orchestration context is unbound, including failed binds.

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
leader checks topology and configures the per-worker contexts. With a leased
CntNotify, each worker samples the READY word before reporting. After accepting
all reports, the leader toggles that word once to publish the configured contexts.
A worker observes the change before accessing its entry in the context array.
Without the lease the AICPU publishes context addresses and READY per worker in GM. Workers bootstrap their graph partitions while the execution gate
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
maps them through `completion_refs`. Each shared Definition child plan holds
the start and count of its fanin indices. A MIX member resolves its instance
once and uses that same node to receive its argument ranges. Roots depend on ENTER. EXIT depends on every sink. The canonical producer
completion gate is the readiness proof for executable and control consumers.

## 6. Bootstrap and Dependencies

Host validates cooperative task shapes and residency limits before publishing
the epoch, participation masks and cooperative ranks. After READY acquire,
each worker validates its context, graph view, epoch, prepared layout, immutable
engine catalog and topology. No Graph child argument binding runs on the device:
Host has already published those arguments and Calls.

For MIX, AICPU clears a run-scoped INIT bitmap before publishing the worker
contexts. After clearing its cluster's SSBUF and executing the publication
barrier, AIV0 sets its cluster bit. AIC and AIV1 observe that bit and execute
the receive barrier before using SSBUF. Each cluster's isolated 128-byte GM
word tracks the subsequent DONE reports. Host seeds its INIT bit as the base
for that accumulation. If the optional INIT bitmap is unavailable, AIV0
publishes INIT through the GM word and its peers confirm it there. Graphs
without MIX use the preseeded GM word. Each worker services bounded control-node scans and prepares its
first execution. AIC and AIV1 then publish their respective DONE bits. Once
AIV0 confirms both bits and completes its own fill, it sets its cluster bit in
a run-scoped CntNotify. AICPU clears this word before publishing any worker
context and opens the DMB launch gate after observing the full participation
mask. Launches without a mapped word use the GM arrival count and final
`bootstrap_complete` publication. A failed
lane contributes no success, and no kernel may enter before this gate opens.

ENTER, EXIT and inline control nodes use bounded scans rather than a ready inbox.
A worker reads each producer's canonical gate and advances only a positive fanin
prefix. When every producer is DONE, a unique CAS changes the control node's
completion gate from its initial state to its completed state. Only the winner
reports the logical completion. There is no control-side fanout/remaining or
wake-list notification path in this protocol.

## 7. Discovery, Dispatch and Completion

### 7.1 Prepared Calls and Execution Tokens

The Host builds a 16-byte `SchedulerExecutionPlan` for every node and one
512-byte immutable `SchedulerPreparedCall` for each physical execution. A Call
names its function, epoch, node, block, lane and complete argument vector.
Tensor arguments are GM descriptor addresses; scalar arguments are value bits.
`args[48]` and `args[49]` name the execution's 128-byte binding, containing its
LocalContext and GlobalContext on separate cache lines. The 50 argument words
end at byte 464. Three 16-byte cache-line spans fill bytes 464–511, which
were alignment padding. The Call stride remains 512 bytes. Calls and bindings have
execution lifetime and remain valid through retirement of the owning slot.

Each execution has a 128-byte control record initialized to BUILT. Its state
identifies the fixed owner and the execution phase. An ordinary block claim
fixes the worker. For MIX, the proxy's block claim and stable SSBUF generation
identify one worker per active lane. Each worker validates its Call and binding. The
claim alone grants no permission to consume business input data.
The worker receives its Host-prepared Call in the same invocation. Cooperative preparation requires an explicit
lane and rejects predicates; ordinary execution tokens retain predicate evaluation.

When a MIX member first accepts a descriptor, it validates the task and caches
the sync-start flag, async-input flag and required issue floor for its lane.
The member reuses those values while waiting for admission. Fanin completion
and the current issue floor are still checked before execution.

The Host first orders tasks by Kahn generation, descending bottom level and
node identity. Flat graphs with zero deferred groups, zero device control
nodes and no predicates are eligible for a local change to this order. For
each wide SPMD task S, the Host estimates how many AICs and AIVs can run
its blocks concurrently. For MIX, it caps the block count at the cluster count
before multiplying by the number of compute lanes of each type. On 32 clusters,
a multi-wave 1C1V task can use at most 32 AIVs. A pure AIV task can use both
vector workers per cluster, up to 64 AIVs. S may be pure or MIX and may request
sync-start.
Each narrow task B considers only the next S needing at least half the workers
on B's engine. An intervening MIX or sync-start task ends the search. If B
and that nearest S fit together, B keeps its position and the search stops
there, even if a wider task follows. If several tasks qualify for S, the Host
moves the nearest B in the original order to the position immediately after S.

B must be an ordinary pure task on that engine, request fewer than half its
workers and have no sync-start requirement. Their combined demand must exceed
the engine's capacity, S must have a greater bottom level, and the two original
generations must differ by at most one. B cannot pass a direct successor.
Each S moves at most one B, and each B can move only once. Cooperative tasks
retain their relative order. The Host verifies that each producer still
precedes its consumer before rebuilding the per-lane issue floors.

The smaller bottom level of B is a structural proxy for flexibility. For
offline validation, compare traces with the same input shapes, kernels and
block configuration. Measure the gap from B's last block completion to the
earliest start of any direct successor, using the end of the measured run for
a sink. For S, compare the earliest and latest block entries and completions
and inspect the intervals during which the tasks overlap on the same engine.
The observed gap can include waits for cores, other inputs and scheduling.
It does not establish how far B can safely move in time. Host binding uses
the static graph and resource demands.

A synchronous kernel completes on return and may signal another rank. A kernel
that needs a signal from another rank must declare deferred completion and
register its wait. Local pipeline waits and synchronization among blocks in a
sync-start cohort follow their existing protocols.

The pure-AIC catalog and the AIV catalog (which includes MIX)
project that order onto the two engines. Each worker advances its private
cursor after confirming that the current task has supplied all its blocks.
Cross-engine predecessors must also have closed their supply before a consumer
can claim a block. The producer's completion gate separately controls when
the consumer can execute. For a dynamically distributed flat executable task without a predicate,
the last valid block claim publishes `claim_seal == logical_block_num`.
This seal proves all ordinary or MIX blocks have owners, so the supply check
can advance the catalog without loading DONE. Native Graph nodes and
predicated tasks still check logical completion, since they can finish without
issuing every block. The consumer checks data dependencies before execution.

For packed sectors, the final valid claimant from each sector reports its bit
after binding. For an ordinary task without sync-start, every reporter that
observes the full mask can store the same terminal block count and supply
seal. Cooperative reporters elect a single publisher for the ordered seal.
Both paths keep data completion separate from supply.

For a flat graph, Host marks runs of three consecutive dynamic ordinary entries
in each engine catalog. When an idle worker reaches a new run, a real claim on
the third entry proves that an issuer already passed the first two supply seals.
The worker can advance directly to the third entry and perform its own normal
claim and data checks. Fixed owners, MIX, predicates and sync-start split runs.
A worker that owes a local execution services it at the usual cursor boundary.

For an exact wave, Host marks the expanded task with `SCHEDULER_TASK_FIXED_OWNER`.
A pure AIC block maps to its cluster index, and a pure AIV block maps to twice
its cluster index plus its vector lane (0 or 1). A 1C1V or 1C2V block maps to
the block assigned to its cluster, where AIV0 publishes the original active mask.
For a full Cube + Vector MIX wave with a power-of-two cluster count of at least eight,
clusters whose index has bit 1 set flip the high index bit to select their block.
All other clusters use their own index. This applies with
either sync_start setting. Fixed assignment satisfies cross-engine supply as
soon as the image is available. An owning core advances its own catalog only
after binding its block or publishing its local MIX descriptor. The cluster
floor, resource order, and actual fanin still guard execution. Per-task dynamic
claim counters and the supply seal stay zero for these tasks.

Each core has one ordinary self-claim slot and one MIX member slot. An
ordinary claim fixes the slot and owner for one block. A MIX member accepts
the descriptor from its cluster's SSBUF bank into its MIX slot. Both slots
retain their identity from preparation through physical return and retirement.
Each slot checks the exact producer's run-local DONE proof before reading its
metadata or binding its completion source. A supply-closed proof still
requires observing completion. Each slot retains its fanin cursor and the
current pending predecessor’s reference and completion source while the
scheduler services other work.
On retry, the worker checks for completion again. It clears the binding
when the cursor advances or the slot is reused. The end of the fanin list
indicates that all predecessors are ready.

Before claiming an ordinary block and again after binding it, a non-publishing
member checks its cluster's SSBUF generation. A new generation naming this
lane gets the usual MIX dependency and order checks. A ready MIX is picked
first. A blocked MIX leaves the ordinary slot available to run its predecessor.
After binding, the ordinary slot keeps its ticket until it resumes. The MIX
member records a dependent ordinary slot at descriptor receipt or when the
ordinary slot first waits on it. While that member is published, its private
flag ends the search after servicing the member. The ordinary slot checks its
remaining dependencies after the member returns. The probe
compares local state and SSBUF fields without polling GM.

Once fanin completes, the worker publishes the payload in the same slot.
A false predicate skips the function and retains its return obligations.
Ordinary selection checks the applicable order before passing the slot to the
executor. Each core keeps its worker ID from the validated run topology. Entry
validates the READY state, the owner and the worker error word.
A MIX member also acquires the SSBUF generation and verifies the run epoch.
The selected worker then indexes the run's fixed Call and Binding tables and invokes the kernel.
L1, L2 and L3 retain the trace cell resolved at READY through identity capture
and retirement. Unsampled L0 sets READY in the private slot. A lane that
has settled the same SSBUF generation can satisfy its own FIN check locally.
An accountant settles only after it publishes the accounting receipt.
When a kernel returns with outstanding deferred conditions, the registering worker
retains its CHECK group and polls the pending entries from its local queue until they settle.

### 7.2 Business Execution and Cooperative Work

The executor calls the prepared function with `Call.args`. The kernel receives
GM arguments and contexts, never a private slot or admission-state address.
Before/after args-dump boundaries coordinate with AICPU's bounded dump service.
Async input observation occurs on the actual consumer after dependencies settle.
The Host aligns the proven access ranges to cache lines and merges overlaps.
The consumer issues at most two single-line DCCI operations followed by one
barrier. A larger footprint or an unproven access uses full-cache maintenance.
A kernel proven to have no such accesses skips both operations.

AIV0 selects an available cluster SSBUF bank, claims one MIX block for execution,
and release-publishes its task, block, rank and lane mask in that bank in the same call. Each named worker accepts a stable generation
once and receives its Host-prepared Call. Execution still requires fanin, ordering and
the applicable sync-start permit. When a participating lane has an earlier
conflicting pure sync-start cohort, every member publishes ARRIVED and waits
for all active arrivals for this generation. Otherwise each member can enter
after its local arrival. Deferred dependencies gate consumers at fan-in, and the
scheduler loop services outstanding async conditions. Each returning member publishes its generation-tagged FIN. When AIV0 needs
the bank again, or drains the run, it checks FIN for every named member.
Protocol capture additionally records an accountant receipt.
The proxy records the last execution ticket published to each bank. At L0–L3 the member
keeps CLAIMED and ENTERED in its handle and publishes RETURNED in GM on physical
return. While the cell is BUILT, Host exports null for entry and ownership. With L4
protocol capture, the member also publishes CLAIMED on receipt and ENTERED at its
kernel call boundary. A timeout can distinguish a member waiting for fan-in from one
that reached the call boundary. Run errors identify the failing operation.

The final MIX entry permit reads the SSBUF generation with an acquire barrier.
A synchronous member can enter the kernel directly after that acquire when
task timing, PMU and argument capture are off. A prebound async group or
possible consumption of an async input keeps the pre-call DDR barrier.

After the kernel returns, its async registration is sealed. A MIX member
records the physical return even if sealing fails. On the CntNotify path each
successful active member publishes its own physical credit P. An async-capable
member also owes one condition credit Q, published at an empty seal or by the
registering worker when its conditions hold. Each worker retains its current
group, pending mask and condition cursor locally. Additional groups use
per-group GM storage in FIFO order. Each pass checks at most eight conditions
from one group and rotates outstanding groups. The worker may run independent
kernels between checks. If the sole pending group has one COUNTER, its first
unsatisfied check retains the validated address and threshold on the worker.
Each pending retry samples that address once. A satisfied sample goes through
the worker error check and descriptor validation before issuing the condition
credit. Enqueuing another group or advancing the head invalidates the cached
address. The resident loop continues its periodic error check between iterations.
Consumers still consult the run-wide activation flag.

When required CntNotify creation exhausts the device quota, the runner releases
idle optional pools in the same context and retries once if it reclaimed capacity.
Live run and capture/replay leases keep their pools throughout the retry.

A5 Host requires all requested CntNotify
slots to be mapped and 32-byte aligned before publishing the image. A missing
slot or invalid mapping fails preparation. On CPU/sim the GM accountant waits for
every active member's successful FIN and contributes the active lane count in
one update. Both paths retire the bank only after the successful FIN check.

### 7.3 Canonical Completion and Retirement

On A5, an ordinary ticket owner cleans its RETURNED line on physical return.
The fence before the CntNotify credit publishes that line with the kernel's writes.
An error fences the return before releasing the lane. Protocol capture keeps its
separate publication boundary. The unprofiled resident variant omits the timing
bookkeeping at retirement. Sampled L0 and higher profiling levels retain it.

On the GM path, `completion_state` encodes the immutable total in bits 63:48,
remaining computation in bits 47:32 and debt in bits 31:0. The initial state
for T physical executions is `(T,T,T)`. A contribution of K physical returns
subtracts `K * ((1 << 32) + 1)`. The first nonempty async group adds one marker
to the debt. Once all physical returns and pending groups settle, the closing
or final-settlement operation grants one FINISH owner, which publishes DONE
inline by changing `(T,0,1)` to `(T,0,0)`. A counted task reaches DONE when
its P and Q contributions reach its CntNotify threshold.

Host records the live nodes with no live consumer after control folding and
Graph expansion. After bootstrap, AICPU observes each sink's existing ADD
threshold, SET notification or GM completion state. A completed sink stays
complete. All sinks must settle before EXIT, including any asynchronous debt.
Independent terminal branches are checked separately. Host-only graphs have
an empty sink list and still go through bootstrap and worker retirement.

A long chain can progress while its sink stays pending. Only after the
watchdog has seen no progress for half its budget does it sample intermediate
state. Profiled retirement snapshots mapped counters for diagnostics. L4
derives its resolved task count from the retired image and exports JSON `null`
if a device task uses a counter without a valid retirement snapshot.

On an error, a worker cancels only its provably owned unstarted handles to FAILED;
it cannot invent a RETURNED Call, computation contribution or group-finished ACK.
Shared armed work is revoked by the failed run's all-core retirement fence
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

A5 HBG L3 traces name the scheduler functions that claim blocks, observe and
publish completion, service MIX and asynchronous work, and decode Graph inputs.
Identity capture, trace staging and deferred idle publication have their own
intervals. Completion and CHECK intervals end when the corresponding calls
return. A parent interval includes its children. Its remaining time includes
control flow, ordinary memory access and instrumentation. See the
[Scalar trace format](../../../../../docs/dfx/chip-swimlane-profiling.md#a5-hbg-scalar-detail-at-level-3).

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
