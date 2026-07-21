# Global DFX Backpressure — block-on-contention + dual-signal freeze

The shared design for how every DFX profiling subsystem (L2Swimlane, PMU,
DepGen, ArgsDump, ScopeStats, tensor_dump) reacts when the host collector
cannot keep the device-side buffer pool refilled. It replaces the older
"bounded wait, then drop the record" model with a **resident block-on-contention
freeze**: an AICPU writer that runs out of ready-queue space or free buffers
parks at its buffer-switch gate until the host clears the contention, rather
than dropping profiling data.

Read alongside the code it documents:

- `src/common/platform/include/common/dfx_backpressure_device.h` — the shared
  `DfxBackpressureHeader` layout and the AICPU-side primitives.
- `src/common/platform/include/aicpu/profiler_device_engine.h` — the device
  producer control flow that calls those primitives (`wait_for_ready_queue_space`,
  `wait_for_free_queue_entry`, `enqueue_ready`).
- `src/common/platform/include/host/profiler_base.h` —
  `ProfilerAlgorithms::update_backpressure_freeze`, the single host-side freeze
  state machine.
- Capacity sizing that keeps this path cold in normal runs:
  [dfx-buffer-capacity-audit.md](dfx-buffer-capacity-audit.md).

## Why block instead of drop

The device writers sit on the AICPU scheduler's critical path. The previous
model gave them a short bounded wait for a ready-queue slot or a free buffer and,
on expiry, counted a dropped record and moved on. That keeps the workload moving
but silently loses profiling data exactly when the system is most interesting
(contended). For DFX correctness we prefer to **stall the producer and lose no
records**, provided the stall cannot deadlock and cannot hang forever on a host
crash. The mechanism below delivers that: resident blocking, bounded only by a
30-second host-crash backstop, with no opt-out.

## The two gate classes (issue #997)

An AICPU writer only ever blocks at one of two buffer-switch gates, so the
coordination is split into two independent freezes — one per gate class:

| Gate | Blocks when | Device signal | Host freeze |
| ---- | ----------- | ------------- | ----------- |
| **push** (`rq_*`) | the per-thread ready queue is full (`next_tail == head`) | `rq_contended` | `rq_freeze_active` |
| **pop** (`fq_*`) | the free queue is empty (`head == tail`) | `fq_contended` | `fq_freeze_active` |

`*_freeze_active` are host→device; `*_contended` are device→host leader signals
that the host consumes and clears. All four are `volatile uint32_t` 0/1 flags in
the `DfxBackpressureHeader` that every subsystem's `DataHeader` embeds once
(physically per-subsystem: each subsystem has its own shared-memory region, so
swimlane's freeze is independent of PMU's).

Splitting by gate class is what makes the freeze **common-mode**: every lane
that reaches a given gate parks at the same gate, producing one lane-aligned gap
in the swimlane instead of one lane sparsifying and skewing a bottleneck read.

## Device side

Three primitives in `dfx_backpressure_device.h`, driven from
`profiler_device_engine.h`:

1. **Leader signal** — `mark_rq_contended` / `mark_fq_contended` set the
   `*_contended` flag once per wait (guarded by a local `signalled` bool). The
   host observes the sticky flag on its next mgmt tick and opens the matching
   freeze.
2. **Peer freeze barrier** — `push_freeze_barrier` / `pop_freeze_barrier` spin
   while the host holds the freeze open (`*_freeze_active != 0`), relaxing via
   the repo-wide `SPIN_WAIT_HINT()` (a no-op on onboard silicon, `sched_yield()`
   in sim). Each carries an acquire `rmb()` after the gate so the host's
   pre-release queue writes are visible to the reads that follow.
3. **Leader park** — `wait_for_release` is used only by a leader whose own
   reclaim depends on the host completing a full open→drain→release cycle on the
   free-queue side (only tensor_dump's arena barrier today). It spins on the
   **disjunction** `fq_contended || fq_freeze_active`.

The push and pop loops in the engine bridge the host round-trip: raising
`*_contended` does not itself park the lane — until the host opens the freeze the
barrier sees `*_freeze_active == 0` and returns at once, so the loop re-checks
the queue and picks up a slot the instant one appears. Once the host has opened
the freeze, the barrier holds the lane until release.

### Timeout backstop

Every barrier and contention spin is bounded by
`Module::kBackpressureWaitCycles`, which every subsystem points at the single
`PLATFORM_DFX_BACKPRESSURE_TIMEOUT_CYCLES` constant
(`PLATFORM_PROF_SYS_CNT_FREQ * 30`, i.e. a 30-second wall-clock ceiling that
scales per arch — a2a3 50 MHz, a5 1000 MHz). The budget exists **only** to break
an infinite spin on host crash or hardware failure; it is not a normal-path
wait. When it expires the writer breaks to its single failure exit
(`return false`), and the caller (`switch_buffer`) accounts the dropped records.
A 30-second expiry therefore means "the host is gone", not "the buffer was
briefly full".

## Host side — the freeze state machine

`ProfilerAlgorithms::update_backpressure_freeze` runs once per tick of the
single `mgmt_replenish_loop`. It must be single-writer for `*_freeze_active`,
which is why it lives on the one replenish thread, not the (possibly sharded)
drain threads. It is idle at zero cost until a lane raises `*_contended`.

**Open (per gate, independent).** If a gate's freeze is closed and its
`*_contended` reads non-zero, set `*_freeze_active = 1`, publish it, then clear
`*_contended`. For the pop gate the freeze is opened *before* `fq_contended` is
consumed so the disjunction `wait_for_release` spins on stays continuously true
— see the escape-window note below.

**Release (conjunction, both gates together).** While either freeze is open,
release **both** only once:

1. the ready queue is fully drained (`queue_heads[q] == queue_tails[q]` for every
   thread), **and**
2. the free queues are refilled to their attainable initial upper limit
   (`replenish_free_queues` returns `pushed == 0`), **and**
3. the per-subsystem `backpressure_release_ready()` predicate holds.

Order matters: RQ-empty is checked first, because once the ready queue is empty
the drain threads are idle and it is safe for the replenish thread to be the sole
free-queue writer (drain-driven `top_up_free_queue` is suppressed while either
freeze is open). Resuming into RQ-empty + FQ-at-limit is one clean common-mode
gap with no immediate re-stall.

### Per-subsystem release predicate

`backpressure_release_ready()` is a CRTP hook, default `true`. A subsystem whose
collector owns a separate, independently-overwritten region must override it to
hold the release until that region is drained — otherwise the device resumes and
overwrites not-yet-pulled data. Only **tensor_dump** does this today: RQ-empty
alone does not imply its payload arena has been pulled. The hook is evaluated on
the replenish thread inside the freeze loop, so overrides must be cheap,
non-blocking, and read only atomics.

## Two correctness arguments

**No (0,0) escape window (pop gate).** `wait_for_release` spins on the
disjunction `fq_contended || fq_freeze_active`. If the host cleared
`fq_contended` before opening `fq_freeze_active`, a leader could observe both
zero mid-transition and escape the park prematurely. Opening the freeze *before*
consuming the contention keeps the predicate continuously true from
`mark_fq_contended()` through release.

**Deadlock-free at any pool size.** The conjunction release cannot wedge
regardless of how small the pool is:

- RQ-drain is host-driven and independent of any buffer a lane holds at a push
  gate — the host can always empty the ready queue.
- "FQ refilled" is defined as `pushed == 0` against the *attainable* initial
  limit `min(kSlotCount, BUFFERS)`, not an unreachable exact-`kSlotCount` fill.

So even a one-buffer-per-pool stress shape drains and releases. The AICPU writer
also publishes its full buffer to the ready queue *before* trying to acquire a
replacement, so the host can always observe the full buffer and return a
recycled one.

## Memory ordering across the PCIe boundary

`*_freeze_active` (host→device) writes go through `write_range_to_device` and
`*_contended` / `queue_tails` (device→host) reads through `read_range_from_device`
so **a5 (non-SVM)** sees them; **a2a3 (SVM)** short-circuits both to no-ops
because the header already lives in shared device memory. Device-side barriers
(`rmb()` after a gate, `wmb()` before publishing a freeze/tail) pair with the
host's `wmb()` before each `write_range_to_device` so a released lane never reads
a stale queue.

## What this observably does under stress

When the pool is deliberately shrunk until the host cannot keep up, the freeze
fires repeatedly and the AICPU scheduler parks at the pop gate. Because the
scheduler is parked, in-flight tasks whose `FIN` it has not yet polled have their
recorded `finish_time` pushed out to the freeze-release instant — i.e. the freeze
**inflates the very `finish_time` it is collecting**. Task *dispatch→finish*
durations in the swimlane are then a measurement artifact of the freeze, not real
kernel time; the unaffected AICore-side `start`/`end` timestamps remain the
ground truth for kernel duration. This is expected: the mechanism trades producer
stall (and skewed AICPU-observed timing) for zero record loss, and normal-sized
pools keep the path cold so it never triggers in production runs.
