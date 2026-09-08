# DFX Backpressure — per-lane block-on-contention

The shared design for how every DFX profiling subsystem (ChipSwimlane, PMU,
DepGen, ArgsDump, ScopeStats) reacts when the host collector cannot keep the
device-side buffer pool refilled. An AICPU writer that runs out of ready-queue
space or free buffers **spins at its own buffer-switch gate** until the host
drains or refills *that lane's* queue, rather than dropping profiling data.

Recovery is per-lane throughout: nothing stops a peer lane, and no host-device
handshake is involved. A stall shows up as one sparse row in the DFX output, for
the duration the host was behind on that row.

Read alongside the code it documents:

- `src/common/platform/include/aicpu/profiler_device_engine.h` — the device
  producer control flow: `wait_for_ready_queue_space`, `wait_for_free_queue_entry`,
  `enqueue_ready`.
- `src/common/platform/include/host/profiler_base.h` —
  `ProfilerAlgorithms::process_entry` / `retry_short_site` (the drain-side refill)
  and `ProfilerBase::mgmt_drain_loop` / `mgmt_replenish_loop`.
- Capacity sizing that keeps this path cold in normal runs:
  [dfx-buffer-capacity-audit.md](dfx-buffer-capacity-audit.md).

## Why block instead of drop

The device writers sit on the AICPU scheduler's critical path. The original model
gave them a short bounded wait for a ready-queue slot or a free buffer and, on
expiry, counted a dropped record and moved on. That keeps the workload moving but
silently loses profiling data exactly when the system is most interesting
(contended). For DFX correctness we prefer to **stall the producer and lose no
records**, provided the stall cannot deadlock and cannot hang forever on a host
crash.

**There is no opt-out.** A caller who would rather accept scattered record loss
than a producer stall has no flag to set today; the drop path survives only as
what happens when the 30-second backstop expires. This is worth
knowing about because it has been mis-stated: PR #1313, which introduced the
mechanism, describes an `enable_dfx_backpressure` flag defaulting to *off*
("drops as before"), but no such flag ever existed in the merged code — commit
`6cea23a1b` shipped block-on-contention unconditionally. The divergence from
`6cea23a1b` shipped block-on-contention unconditionally. The divergence from the
opt-in requested in #997 was never a recorded decision. Adding the gate is
tracked separately; see #2147.

## The two gates

An AICPU writer only ever blocks at one of two buffer-switch gates:

| Gate | Blocks when | Recovered by |
| ---- | ----------- | ------------ |
| **push** | the per-thread ready queue is full (`next_tail == head`) | the one drain shard that serves that ready queue advancing its head |
| **pop** | this lane's free queue is empty (`head == tail`) | that same drain shard publishing a buffer into this lane's free queue |

Both recoveries are the work of a single host thread acting on a single queue, so
neither gate needs a peer lane to stop or a signal to be exchanged. The writer
simply re-reads its own queue until the slot appears.

### Timeout backstop

Each gate is bounded by `Module::kBackpressureWaitCycles`, which every subsystem
points at the single `PLATFORM_DFX_BACKPRESSURE_TIMEOUT_CYCLES` constant
(`PLATFORM_PROF_SYS_CNT_FREQ * 30`, i.e. a 30-second wall-clock ceiling that
scales per arch — a2a3 50 MHz, a5 1000 MHz). It is not a normal-path wait: it
exists to break a spin the host will never end. Expiry says only that this queue
saw no host progress for the whole budget — a dead host, a wedged drain thread,
or a pool too small to ever satisfy this lane — so it is a signal to investigate,
not proof the host is gone.

**One budget spans the whole wait**, taken once before the loop rather than
re-armed per attempt. That distinction is load-bearing: a budget that restarts on
every iteration can never expire, so a lane whose pool is permanently short would
spin forever instead of reporting a drop.

Both gates fail only while ownership is still on the device — a full ready queue
*before* publication, or an empty free queue while claiming a replacement — so the
caller can safely account the affected records dropped. The push gate runs
entirely before the ready-queue tail advances, so publishing and succeeding are
atomic: there is no state in which a buffer has been handed to the host and the
enqueue still reports failure.

## Host side — the drain shard owns every free queue

Each ready queue *q* is served by exactly one drain shard
(`mgmt_drain_loop(queue_start, queue_stride)` strides `q` by the shard count), and
every `free_queue` an entry on *q* resolves to therefore belongs to that one
shard. **That shard is the sole runtime writer of those queues.** The replenish
thread's runtime writes go only to host-side recycled lanes; it never publishes
into a device free queue. Single-writer here is a property of the structure, not
of timing.

Refill happens in two places on the owning shard:

1. **After each drained entry** — `process_entry` tops up the free queue the
   entry came from, out of the shard's own recycled lane. This is the steady-state
   path and it allocates nothing.
2. **After each sweep, for lanes that came up short** — when the recycled lane
   runs dry mid-top-up, `process_entry` reports the site and the drain loop
   retries it on every subsequent sweep until it fills.

The second is what makes per-lane recovery work at all. The first is
*entry-driven*, and a lane that has run dry holds no buffer, so it has nothing
left to publish and would never trigger its own refill. The retry list is empty
whenever nothing is starved, so the path costs nothing in a normal run.

Retrying after **every** sweep rather than only on an idle one matters when a
shard serves several lanes: a shard busy draining one lane would otherwise never
reach an idle sweep, and its starved sibling would wait behind unrelated traffic.

`replenish_free_queues` still fills every instance to `kSlotCount`, but only from
`proactive_replenish` at startup, before any drain thread exists. That is also
what makes its `shard_index = -1` mode safe: only there may `obtain_buffer`
allocate, and only there may `pop_recycled_for_startup` consume from every
shard's recycled lane. At runtime each recycled lane stays strictly SPSC —
replenish thread produces, owning drain shard consumes.

## Per-subsystem arena acknowledgement

`publish_arena_acks()` is a CRTP hook, default a no-op, called once per replenish
tick. A subsystem whose collector owns a separate reusable region overrides it to
publish how much of that region the host has consumed, per lane.

Only **args_dump** does this today. Each AICPU thread has its own payload arena,
and `arena_write_offset` is a monotonic cursor written modulo `arena_size`, so a
write that lands on or straddles a physical wrap would overwrite bytes the host
may not have pulled. `ensure_dump_arena_capacity` seals and publishes the
in-flight meta buffer, snapshots its own `published_payload_count`, and waits for
its own `completed_payload_count` to reach it. The host advances that counter for
thread *t* once `args.bin` has accepted every payload thread *t* published — and
independently of every other thread, because thread *t*'s arena is thread *t*'s
alone. The hook is evaluated on the replenish thread, so overrides must be cheap
and non-blocking.

## Deadlock-freedom

A starved lane always recovers, at any pool size, along this chain:

1. The lane published its full buffer to the ready queue *before* trying to
   acquire a replacement, so the host can always observe it.
2. Its drain shard pops that entry, hands the buffer to the collector, and — if
   its recycled lane was dry — records the site for retry.
3. The collector finishes the buffer and notifies it done.
4. The replenish thread drains the done queue into a recycled lane and tops
   recycled lanes up to their watermark, allocating if needed. Neither step waits
   on any device state.
5. The shard's next sweep retries the site and publishes the buffer into the
   starved lane's free queue.

Step 4 must land the buffer in the recycled lane of the shard that needs it,
since a drain shard pops only its own lane (`pop_recycled(kind, shard)`) and
cannot steal from a sibling. `drain_done_into_recycled` routes by
`select_recycled_shard(kind, origin_shard)`, where `origin_shard` is the collector
shard that consumed the buffer — the same shard that drained the starved lane's
entry — and origin wins deficit ties. A starved shard also has the largest
deficit by construction, so it wins outright when watermarks are in play.

## Memory ordering across the PCIe boundary

Host writes into device-visible queue fields (`free_queue.tail`,
`buffer_ptrs[]`, `queue_heads[]`) go through `write_range_to_device` and
device→host reads through `read_range_from_device`, so **a5 (non-SVM)** sees
them; **a2a3 (SVM)** short-circuits both to no-ops because the header already
lives in shared device memory. A bulk host→device write-back is deliberately
avoided — it would roll back AICPU-owned fields. Device-side barriers (`rmb()`
after taking a queue slot, `wmb()` before publishing a tail) pair with the host's
`wmb()` before each `write_range_to_device`.

## What this observably does under stress

When the pool is deliberately shrunk until the host cannot keep up, the AICPU
scheduler parks at the pop gate for the lane that ran dry. Because the scheduler
is parked, in-flight tasks whose `FIN` it has not yet polled have their recorded
`finish_time` pushed out to the moment the lane recovers — i.e. the stall
**inflates the very `finish_time` it is collecting**. Task *dispatch→finish*
durations in the swimlane are then a measurement artifact of the stall, not real
kernel time; the unaffected AICore-side `start`/`end` timestamps remain the
ground truth for kernel duration. This is expected: the mechanism trades producer
stall (and skewed AICPU-observed timing) for zero record loss, and normal-sized
pools keep the path cold so it never triggers in production runs.

**The artifact is confined to the lanes that actually stalled.** That is the
deliberate trade against #997, which asked for the opposite: *"The DFX result
shows one contiguous blank window per stall"* — one lane-aligned gap, produced by
stopping every lane at its gate so a reader could not mistake one sparse row for
a real bottleneck. Per-lane recovery gives that alignment up. It buys back the
cost of the alignment, which was that a single dry lane perturbed every row of
the measurement. Reading a swimlane taken under pool pressure now means checking
which rows stalled rather than trusting that a gap is common-mode; the AICore
timestamps are unaffected either way.
