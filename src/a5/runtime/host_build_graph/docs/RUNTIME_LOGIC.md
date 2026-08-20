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

An orchestration fatal stops this sequence and is propagated through
`orch_error_code`.

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
| `PTO2TaskDescriptor` | Full task ID, kernel IDs, packed-buffer addresses |
| `PTO2TaskPayload` | Tensors, scalars, predicate, local-ID fanins, dispatch metadata |
| `PTO2TaskSlotState` | Active mask, attributes, block/subtask counters, completion state, task/payload bindings |

The host/device boundary is POD and position-independent. Fanins are integer
producer IDs, not pointers, and a slot state names its payload and descriptor by a
delta from its own address — so the image needs no fixup on either side of the
copy.

### 3.1 What Ships: the Arena's Three Zones

Three rules decide every byte of the runtime arena:

1. **Whoever generates a value writes it.** Content the host generates is written
   on the host and copied down. Content that is a function of the *layout* rather
   than of the *run* is written by the side that reads it.
2. **A copy carries per-run content, never an initialization pattern.** Shipping
   bytes the device could derive from the layout spends link bandwidth
   transporting a constant.
3. **Initialize once.** A region whose content does not differ between runs is
   established once, not re-established per bind.

They partition the arena into three contiguous zones, and `runtime_reserve_layout`
reserves them in this order:

| Zone | Regions | Copied | Allocated on device | Written by |
| ---- | ------- | ------ | ------------------- | ---------- |
| device-only | `sm_handle`, the completion mailbox, `PTO2SchedulerState` and its thirteen queue slot arrays | never | yes | AICPU at boot |
| copied | `[off_copied_begin, off_copied_end)`: the runtime header | whole zone, one copy | yes | host |
| host-only | orchestrator block: `fanin_seen_epoch` / `scope_tasks` / TensorMap, ~9.3 MB | never | **no** | host, during graph construction |

The copied zone is the fixed device prefix's final zone. After orchestration, its
end is followed by the compact shared-memory image and then the Graph Definitions
and submissions. `bind` stages those three run-generated regions contiguously and
uses one `copy_to_device` starting at `off_copied_begin`.

**Why the host-only zone comes last.** No device code dereferences a pointer into
the orchestrator block — hbg has no device-side orchestrator, and the one
orchestrator field the scheduler reads, `inline_completed_tasks`, is a scalar in
the runtime header. Putting the block at the tail lets `setup_static_arena` request
only `layout.device_bytes`, the copied + device-only prefix, so those ~9.3 MB are
never reserved on the device at all. `runtime_wire_host_only_pointers` is therefore
host-only, and `runtime_clear_host_only_pointers` drops those pointers before the
copied zone is uploaded so no host address crosses the boundary.

**Why the scheduler state is device-written.** `PTO2SchedulerState` holds no
per-run content: `sm_header` and the ring pointer derive from a pooled SM base,
queue capacities are compile-time constants, and hbg never advances
`last_task_alive`. Its one host-side entry point, `on_scope_end`, is an empty stub
here. So the host would only be writing an initialization pattern — 203,392 bytes
of it, dominated by `AsyncWaitList::entries` — for the device to receive and never
read. `PTO2Runtime` therefore holds a *pointer* to it, wired from
`off_scheduler` on each side, and the AICPU calls `init_data_from_layout` at boot.

**Why the queue slots are device-written.** `push` claims `slots[pos & mask]` only
when that slot's `sequence` already equals `pos`, so an empty queue is a
`0..capacity-1` ramp, not zeroed memory: on zeroed slots position 0 happens to
match and every later position reads a lower sequence, which is the full-queue
signal, so such a queue accepts one push and then reports full. The ramp is
mandatory but it is a function of `capacity` alone, so
`PTO2SchedulerState::seed_queue_slots()` writes it on the device rather than `bind`
shipping 1,775,616 bytes of it. The ready queues are still *not* bounded to
`total_tasks`: graph execution expands a GRAPH task into on-device nodes that push
past the host task count, so every slot must carry a valid sequence.

Both run before the boot thread publishes `runtime_init_ready_`, which is what
releases the peer threads into the dispatch loop, so no push can observe an
uninitialized queue.

**Boot cost, not per-run cost.** The sequence invariant is lap-relative: a free
slot's sequence tracks the position it serves, and `pop` releases a slot with
exactly the value the next lap's `push` expects. A drained queue is therefore
already an empty queue, which is why `tensormap_and_ringbuffer` can leave
`PTO2ReadyQueue::reset_for_reuse()` empty and never touch the positions. hbg
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

The shared-memory mirror is sized to ring capacity (task window) but a run only
writes `[0, total_tasks)`, and the device boots scheduler-only and reads no SM slot
past `total_tasks`. So the SM H2D shipped each run is bounded, not capacity-sized —
the contract that keeps `bind` proportional to the workload.

The header is zeroed on the host; `descriptors`, `slot_states` and
`completion_flags` are each written per task at submit. Payloads come from one
`TaskPayloadSpace`: every slot stores a self-relative reference to its record, so
Graph tasks can carry more tensors or scalars than an ordinary fixed payload.
Per-slot reset is init-on-write in `orch::prepare_task` as each slot is claimed —
there is no window-wide reset.

At the end of orchestration, compaction restacks the descriptor prefix, the exact
live TaskPayloadSpace byte range, the slot-state prefix, and the completion flags.
It preserves each slot's payload offset while rebinding its relative references.
That compact SM image follows the copied runtime header and precedes the Graph
block in one staging buffer, so all host-generated runtime data travels in one
H2D copy.

The heap is committed first because host construction needs its device addresses.
The runtime arena is not committed until the compact SM and Graph block sizes are
known; the arena bank then allocates it once or reuses a prior allocation whose
capacity is sufficient.

## 4. Whole-Graph Capacity

The runtime uses one task ring, one graph heap, and one TensorMap pool. They are
capacity-bounded storage, not streaming flow-control buffers:

- the task ring and the graph heap are forward-only bump allocators;
- task slots and heap bytes are never recycled mid-run; and
- TensorMap entries are held for the whole run.

There is no reclaim channel from the scheduler back to the allocator, so the
allocators carry no reclaim pointer and no back-pressure wait.

`completed_watermark` records the contiguous prefix of completed device tasks.
It supports completion/consumer metadata only; it does not reclaim the task ring
or heap.

There is no post-run sweep that makes graph space reusable. Runtime destruction
releases the complete arena, and the next run starts from a newly initialized
image.

### 4.1 Allocation Failure

The graph must fit the configured task window, heap, fanin capacity, and
TensorMap pool. Because nothing is reclaimed, a request that does not fit can
never become satisfiable — the allocator names the exhausted resource and fails
on the spot. There is no wait and no timeout.

Representative allocator output is:

```text
FATAL: Graph Heap Exhausted!
The whole graph must fit the configured ring; nothing is reclaimed mid-run.
  Task window: used=.../...
  Graph heap:  used=.../..., available=...
  Requested:   ... bytes + 1 task slot
```

This is host-orchestration logging. The allocator records the corresponding
runtime error and unwinds; it does not terminate the process directly.

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
2. Explicit and discovered producers are deduplicated into
   `fanin_local_ids[]`.
3. OUTPUT/INOUT regions register the new task as producer.
4. Each producer tracks its highest consumer local ID for completion metadata.

There is no fanout adjacency or dependency pool. A per-slot completion flag is
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

Early producer propagation is currently disabled in HBG. The shared scheduler
retains early-staging code for parity with `tensormap_and_ringbuffer`, but HBG's
boot classifier and wake lists are the active readiness path.

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
- Completion sets the task's flag, advances the contiguous
  `completed_watermark`, and reclassifies its wake-list consumers.

The drain's `pending_task` stays valid for the complete attempt: all participant
threads load it before the coordinator can pass the stage-done barrier and clear
it. A recovery return for a null pointer would describe an unreachable state and
could strand the drain protocol, so the active path relies on that invariant.

## 8. Scalar Access During Construction

`get_tensor_data` and `set_tensor_data` operate on registered host views of
external tensors. They cannot wait for a submitted device producer because the
device scheduler starts only after orchestration returns. Runtime-created graph-
heap outputs also have no host view.

Producer references are checked against the complete bound descriptor ID before
a slot is used, preventing masked-slot aliasing. See
[SCALAR_DATA_ACCESS.md](SCALAR_DATA_ACCESS.md) for the supported contract.

## 9. Errors and Diagnostics

The runtime latches orchestration and scheduler errors in shared memory and maps
them to the negative run status observed by the host. Important validation paths
include:

- invalid arguments (`-5`);
- sync-start residency violations (`-7`);
- tensor wait timeout (`-8`); and
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
