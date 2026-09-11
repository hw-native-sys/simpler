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
device: attach the image, classify tasks, dispatch with AICPU schedulers
        ↓
host: collect outputs and destroy/reset per-run state
```

The device has no orchestration thread. Every launched AICPU thread participates
in scheduling its assigned AICore workers; the highest-index thread first
attaches the prebuilt runtime and publishes the boot barrier.

This ordering is the defining constraint of the runtime. The host constructs the
whole graph before any device task can complete.

## 2. Callable and Run Lifecycle

### 2.1 Registration

`register_callable_impl` materializes the orchestration bytes as a temporary
shared object, opens it with `dlopen(RTLD_LOCAL)`, and resolves:

- `aicpu_orchestration_config`;
- `aicpu_orchestration_entry`; and
- `framework_bind_runtime`.

The resolved handle and entry points belong to the registered callable and stay
alive across prepared runs. Unregister/context teardown closes the handle and
removes its temporary file.

Child AIC/AIV callables are uploaded separately. Their resolved device function
addresses populate the per-run dispatch table.

### 2.2 Host Graph Construction

For each run, the host:

1. validates and stages external tensors into a run-owned host accessor;
2. reserves one backing arena for runtime/shared-memory subregions;
3. binds the runtime to the orchestration DSO;
4. calls the orchestration entry synchronously;
5. finalizes task counts and the graph image; and
6. copies the shared-memory image and the arena's copied zone to the device.

An orchestration fatal stops this sequence before the upload. The orchestrator
runs on the host, so its code is latched in `OrchestratorState::fatal_code` and
never reaches shared memory; the bind maps it onto the status the caller sees.

Step 1 stages each caller tensor through the runner's retained temporary buffer
rather than a per-run `device_malloc` / `device_free` pair: bind packs the run's
non-child tensors to a 1024-aligned required size, grows the buffer only when a
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

The H2D copy of a staged tensor precedes its registration with the run's host
accessor, so a reused slice can never expose the previous run's bytes to
orchestration.

A pure `OUT` tensor is staged into a slice but never copied in, and nothing
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

The boot thread attaches the already-populated arena without resetting it. All
threads classify/dispatch their core partitions and then shut those cores down.
The last arriving thread destroys the attached runtime before publishing cleanup
eligibility. Exactly one returning AICPU thread claims that eligibility and
resets executor/scheduler state for the next run.

Publishing cleanup only after destruction prevents `deinit()` from racing the
runtime arena or this run's host accessor.

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

### 3.1 What Ships: the Arena's Two Zones

Three rules decide every byte of the runtime arena:

1. **Whoever generates a value writes it.** Content the host generates is written
   on the host and copied down. Content that is a function of the *layout* rather
   than of the *run* is written by the side that reads it.
2. **A copy carries per-run content, never an initialization pattern.** Shipping
   bytes the device could derive from the layout spends link bandwidth
   transporting a constant.
3. **Initialize once.** A region whose content does not differ between runs is
   established once, not re-established per bind.

They partition the arena into two contiguous zones, and `runtime_reserve_layout`
reserves them in this order:

| Zone | Regions | Copied | Allocated on device | Written by |
| ---- | ------- | ------ | ------------------- | ---------- |
| device-only | `sm_handle`, the completion mailbox, `SchedulerState` and its thirteen queue slot arrays | never | yes | AICPU at boot |
| copied | `[off_copied_begin, off_copied_end)`: the runtime header | whole zone, one copy | yes | host |

The copied zone comes last, so `bind` is a single contiguous `copy_to_device`
starting at `off_copied_begin` — and the device's shared-memory tail begins
exactly where that zone ends. Both bounds are layout fields, so no consumer infers
a boundary from which region happens to be reserved first —
`bind_callable_to_runtime_impl` asserts only that they are ordered and in range.

**Why the orchestrator is not in the arena at all.** hbg has no device-side
orchestrator, so nothing on the device reads its state: not the `fanin_seen_epoch`
table, not the scope arrays, not the TensorMap (~9.3 MB between them). It is
therefore a plain host object that owns those arrays — `OrchestratorState::init`
allocates them — and `RuntimeContext` reaches it through a pointer that `bind` drops
before the copied zone is uploaded, so no host address crosses the boundary. A
`static_assert` keeps `RuntimeContext` trivially copyable, which is what forbids
putting an owning member back inside it. The one orchestrator value the device-side
scheduler reads, the count of tasks completed inline during orchestration, is a
scalar `rt_orchestration_done` publishes into the runtime header.

**Why the scheduler state is device-written.** `SchedulerState` holds no
per-run content: `sm_header` and the task-header pointer derive from a pooled SM base,
queue capacities are compile-time constants, polling reserves no wiring or
dependency pool (readiness comes from the task table's `task_states`, which
the task header owns), and it has no host-side entry point at all. So the host would
only be writing an initialization pattern — 203,392 bytes
of it, dominated by `AsyncWaitList::entries` — for the device to receive and never
read. `RuntimeContext` therefore holds a *pointer* to it, wired from
`off_scheduler` on each side, and the AICPU calls `init_data_from_layout` at boot.

"Never read" holds for `entries` itself, not for the scalars that index it. The
region arrives holding the pooled allocation's previous generation, so every
field the dispatch loop reads before anything writes it needs a value from
`init_data_from_layout`: the queue headers, and `AsyncWaitList`'s `busy` and
`count`. A residual `count` is the length the resolution thread's poll walks
`entries` by, so it reads past the region and faults on an address that belongs
to no mapping (issue #2121); a residual `busy` is the same miss inverted, making
every drain a no-op and stranding deferred completions.

**Why the queue slots are device-written.** `push` claims `slots[pos & mask]` only
when that slot's `sequence` already equals `pos`, so an empty queue is a
`0..capacity-1` ramp, not zeroed memory: on zeroed slots position 0 happens to
match and every later position reads a lower sequence, which is the full-queue
signal, so such a queue accepts one push and then reports full. The ramp is
mandatory but it is a function of `capacity` alone, so
`SchedulerState::seed_queue_slots()` writes it on the device rather than `bind`
shipping 1,775,616 bytes of it. The ready queues are still *not* bounded to
`total_tasks`: graph execution expands a GRAPH task into on-device in-graph tasks that
push past the host task count, so every slot must carry a valid sequence.

Both run before the boot thread publishes `runtime_init_ready_`, which is what
releases the peer threads into the dispatch loop, so no push can observe an
uninitialized queue.

**Boot cost, not per-run cost.** The sequence invariant is lap-relative: a free
slot's sequence tracks the position it serves, and `pop` releases a slot with
exactly the value the next lap's `push` expects. A drained queue is therefore
already an empty queue, which is why `tensormap_and_ringbuffer` can leave
`ChipReadyQueue::reset_for_reuse()` empty and never touch the positions. hbg
re-establishes both on every attach today because the queue *headers* are reset
per bind; the combination to avoid is resetting the positions while leaving the
sequences mid-lap, which makes `push` read a sequence above its position and spin
as if a peer were mid-publish.

**Why the mailbox is device-written, and why zeroing `seq` is not optional.**
`try_pop` bounds its scan with `head` and gates publication on
`entries[t].seq == t + 1`, while a producer bumps `head` *before* it stores `seq`.
So between those two a consumer already sees `t < head` and reads that slot: a
residual `seq` equal to `t + 1` would hand out a message the producer has not
written. `init_empty()` therefore zeroes `head`, `tail` and every slot's `seq`; the
remaining message fields are written before their `seq` and never read ahead of
`head`, so they need nothing. This is the one region whose device-side
initialization is *not* O(1), and only because the cursors restart at zero every
bind — once they persist, positions never repeat, a residual `seq` is always below
`t + 1`, and the whole thing collapses to `tail := head`, which also discards what
an error-aborted run left undrained. `MonotonicSeqSurvivesCapacityWrapWithoutZeroing`
pins the invariant that makes that safe.

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

Completion is published per task, in `task_states[local_id]`, and reclaims
neither task slots nor heap.

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

There is no fanout adjacency or dependency pool. A per-slot progress state is
the readiness truth on device.

## 6. Boot Classification and Wake Lists

Submit does not push tasks into ready queues. After the graph arrives on device,
boot classification scans every submitted task exactly once:

- a task whose fanins are all complete is routed to its shape queue;
- otherwise it registers on its latest-submitted unmet producer's intrusive
  wake list -- the producer likeliest to complete last, which minimises how
  often a waiter is transferred between wake lists and the CAS contention
  those transfers cause; and
- producer completion reclassifies every detached waiter.

Completion flags are monotonic, so this consumer-pull scheme cannot miss a
producer transition and does not require periodic dependency polling.

The dispatchable shapes are `AIC`, `AIV`, and `MIX`; dependency-only `DUMMY`
tasks use a dedicated queue and complete without AICore dispatch.

Early dispatch is detected by the publish list, the wake list's dual keyed on
publication instead of completion. The host qualifies candidates at submit
(at least one producer, every producer flagged and none of them a Graph shell —
a shell has no publication event — no dispatch predicate, dispatchable
shape) and sorts
a candidate's fanin row by ascending local id; a candidate hangs on its
latest-submitted unpublished producer, the producer's publish event seals the
chain (sentinel
exchange) and hands the detached waiters to idle threads, and an all-published
rescan verdict queues the candidate for pre-staging. Release rings the staged
doorbells at the ready funnel (`push_ready_routed`), the moment readiness is
decided. Completion readiness itself remains the boot classifier + wake lists.

## 7. Dispatch and Completion

- AIC/AIV dispatch claims ranges of logical block indices from
  `next_block_idx` and requeues unfinished wide tasks.
- MIX dispatch selects cluster offsets whose used lanes share one valid
  placement. The tracker uses a 128-bit bitset because the flattened offset is
  `cluster * 3`, reaching above bit 63 on supported devices.
- Sync-start cohorts stage locally when possible; wider ownership spans use a
  generation-tagged global drain before launch.
- Every lane completion increments `completed_subtasks`. The task completes once
  that count equals `block_num * popcount(active_mask)`.
- Completion sets the task's flag and reclassifies its wake-list consumers.

The drain's `pending_task` stays valid for the complete attempt: all participant
threads load it before the coordinator can pass the stage-done barrier and clear
it. A recovery return for a null pointer would describe an unreachable state and
could strand the drain protocol, so the active path relies on that invariant.

## 8. Scalar Access During Construction

`get_tensor_data` and `set_tensor_data` operate on registered host views of
external tensors, and only on tensors that no submitted task produces. A producer
— named either by the tensor's `owner_task_id` or by an overlapping TensorMap
entry — fails the call with `INVALID_ARGS`, because the device scheduler starts
only after orchestration returns and nothing it produces exists yet. A runtime
allocation is rejected by the same rule; its graph-heap buffer has no host view
either. See [SCALAR_DATA_ACCESS.md](SCALAR_DATA_ACCESS.md) for the supported
contract.

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
