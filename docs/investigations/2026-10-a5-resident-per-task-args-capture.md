# A5 `host_build_graph` resident scheduler: per-task argument capture

Status: **design only; the capability is deferred to its own change and is not
part of PR #2588.** The resident (AICore Scheduler) route on A5
`host_build_graph` has no per-task argument producer, and the runtime says so:
`args_dump_capture_supported_impl` answers 0 and `simpler_prepare_run` refuses
an `enable_dump_args` request with `PTO_RUNTIME_ERR_UNSUPPORTED` before the
bind, so a caller is told the capture is unavailable rather than handed an
artifact taken at some other instant. That boundary is shipped; what follows is
the bounded design for closing the gap, which nothing in the tree depends on.
Revised five times after review: the first version's mix-task protocol and its
storage figure were wrong; the second's claim that *one argument* is a bounded
service quantum was wrong, because `dump_arg_record` blocks and one argument is
not bounded; the third's single-CAS slot claim was not expressible in the
atomics this scheduler has; the fourth's gate still admitted a late claimer,
truncated the run epoch and task id into one word, published the shared line as
a line, and let two open reservations publish each other's unfinished records;
and the fifth still wrote the slot line back on the abort path, and still
conflated the phase's identity with the per-request sequence its own progress
loop advances. All five are corrected below, and the protocol is **not** claimed
complete beyond the interleavings walked here.

## Why the route has no producer

Every capturing path calls `dump_args_for_task` from the thread handing the
task to a core (`BEFORE_DISPATCH`) or reaping its completion
(`AFTER_COMPLETION`) — `scheduler_dispatch.cpp` and `scheduler_completion.cpp`
in both runtimes. On the resident route the on-core schedulers do both, so no
AICPU thread observes an individual task starting or finishing. The two instants
an AICPU thread does own are the run's, and a read there is not that task's
data:

- A produces X and B consumes it: a pre-run read gives B the content X held
  *before A ran*.
- Several tasks write one buffer in turn — which is what the chained workloads
  do, and what the HBG graph heap does generally — so a post-run read reports
  the *last* write as every one of their outputs.

## What the existing mechanisms already provide

| Piece | Where | Size | Owner |
| ----- | ----- | ---- | ----- |
| Metadata buffers | `DumpMetaBuffer`, free/ready queues | 8 × 256 × 128 B = **256 KiB per lane** | AICPU pops, host consumes |
| Payload arena | per dump lane | 8 × 256 × `PLATFORM_DUMP_AVG_TENSOR_BYTES`(64 KiB) = **128 MiB per lane** | AICPU writes, host acknowledges per payload |
| Back-pressure | `published_payload_count` / `completed_payload_count` | — | host releases, producer waits |
| Selection | `CoreTaskArgs::dump()` → per-task mask; `PARTIAL`/`HYBRID`/`FULL` | — | caller |
| Per-core staging precedent | `PmuAicoreRing` | 2 × 64 B per core | AICore writes, AICPU drains |
| Best-effort hang read | `dump_running_task_outputs` | — | AICPU, explicitly torn/partial; **not a precedent for an exact capture** |
| SSBUF | `SchedulerSsbufRegion` | **exactly its 1024 B budget** (`static_assert(sizeof == SCHEDULER_SSBUF_REGION_SIZE)`); the other 2 KiB per cluster is left to user kernels | scheduler |

The arena, admission, back-pressure, truncation and terminal accounting all
exist and are bounded. What is missing is a producer that can call
`dump_arg_record` at the task boundary. SSBUF cannot host the control words:
its region is allocated to the byte, so both mechanisms place them in GM.

## Mechanism R — task-boundary rendezvous

### Participants and barriers

A task's capture is owned by the set of subtasks its metadata marks active, not
by whichever core gets there first. `SchedulerTaskMetadata::active_mask` is the
authority, and a mix task's subtasks are placed within one cluster, so the
rendezvous state is per *dispatch slot of a cluster*:

- **BEFORE barrier.** Each participating core, after claiming its slot and
  before `execute_task`, sets its bit in the slot's `arrived_before` mask and
  waits. When the mask equals `active_mask` the last arriver publishes the
  slot's request to the AICPU. Every participant then waits for the slot's
  `served_before` seal. **No core executes until the seal is observed**, which
  is what makes the captured input the content the task actually reads — the
  first version's first-requester-wins let a peer subtask run while the input
  was being copied, and is withdrawn.
- **AFTER barrier.** Same shape after `execute_task` and the output
  write-back below: a core sets `arrived_after`, the last arriver publishes, all
  wait for `served_after`, and only then does any of them publish completion.
  One arrival cannot prove the other subtasks finished writing, which is the
  second half of what was wrong before.
- **Record ownership.** The AICPU serves a `(task, phase)` once. The seal is the
  slot's, not the core's, so three subtasks of a mix task produce one record set
  carrying all three `func_ids` — exactly what `dump_args_for_task` already
  stamps from the active set.

### Slot identity, publication and reset, in the atomics that exist

The only atomic widths available are the ones `scheduler_memory.h` provides:
64-bit `scheduler_gm_compare_exchange`, `scheduler_gm_fetch_or` (itself a CAS
loop on AICore), `scheduler_gm_fetch_add`, and `scheduler_gm_store` /
`scheduler_gm_query`, which on AICore are the device-direct `st_dev` / `ld_dev`
rather than cached accesses. **There is no wider transaction**, so a claim
cannot publish an identity and clear the masks in one step.

Three rules follow, and together they are what makes the line safe:

1. **One access discipline for the whole line.** Every access to a slot — the
   claimer's initialization included — is a 64-bit GM atomic or a device-direct
   `scheduler_gm_store` / `scheduler_gm_query`. The line is **never written back
   or invalidated as a line**: `scheduler_publish_cache_line` and
   `scheduler_observe_cache_line` do not appear in this protocol at all. The
   second draft published the line after initialization, which could write a
   stale cached copy of a peer's atomically-updated word back over it; with no
   line operation anywhere, that race has no site. (Tensor payloads still use
   `dcci(CACHELINE_OUT)` — different memory, single writer, no atomics.)
2. **The gate word carries no identity.** `claim` is
   `(state << 62) | occupancy`, where `occupancy` is this slot's 62-bit
   occupancy counter. Identity is **never truncated**: the full 64-bit
   `run_epoch` and the full `TaskId` travel in the request and acknowledgement
   lines, which are 64 B each and have room for them. The second draft packed a
   30-bit epoch and a 32-bit task index into the gate; that is withdrawn.
   `occupancy` is not an identity — it is a reuse counter, zeroed with the array
   at run init and advanced once per claim, so its bound inside a run is the
   run's dispatch count and the 62-bit space is unreachable. Nothing outside the
   slot consumes it.
3. **A claim starts only from a state that proves nobody is inside.** The CAS's
   expected value is the full observed word, and a core attempts it only from
   `FREE` or from `SETTLED` of a *different* occupancy. `CLAIMING` and `READY`
   are never source states, so the second draft's hole — a late participant
   CASing `CLAIMING|g → CLAIMING|g` and re-clearing the masks — cannot be
   reached: that word is not an allowed expected value, and `expected ==
   desired` never occurs because the desired state differs from both allowed
   sources.

#### POD fields and their owners

```c
// 64 B exactly, one per (cluster, pending slot):
//   SCHEDULER_CLUSTER_CAPACITY(36) × SCHEDULER_PENDING_SLOT_COUNT(2) = 72
struct ArgsCaptureSlot {
    volatile uint64_t claim;             // (state << 62) | occupancy
    volatile uint64_t arrived_before;    // participant bitmask
    volatile uint64_t arrived_after;     // participant bitmask
    volatile uint64_t released_before;   // participant bitmask
    volatile uint64_t released_after;    // participant bitmask
    volatile uint64_t publisher_before;  // (core_id + 1) << 32 | phase_seq
    volatile uint64_t publisher_after;   // (core_id + 1) << 32 | phase_seq
    volatile uint64_t aborted;           // 0, or 1 | (site << 8)
};
```

| Field | Written by | Primitive | Read by |
| ----- | ---------- | --------- | ------- |
| `claim` → `CLAIMING` | the claiming core | `compare_exchange`, expected ∈ {`FREE`, `SETTLED`\|other occupancy} | every participant |
| `claim` → `READY` | that same core, last in initialization | `scheduler_gm_store` | every participant |
| `claim` → `SETTLED` | **the last participant to leave the AFTER phase** — the one whose `fetch_or` completed `released_after == active_mask` | `scheduler_gm_store` | the next claimer |
| `arrived_before` / `arrived_after` | each participant, its own bit | `fetch_or` | the arrivers; read-only to the AICPU |
| `released_before` / `released_after` | each participant, its own bit, on leaving that phase | `fetch_or` | the arrivers |
| `publisher_before` / `publisher_after` | **that phase's last arriver, once** | `scheduler_gm_store` | every peer of that phase, and the AICPU |
| `aborted` | the first waiter to time out | `compare_exchange(0 → 1 \| site)` | peers, the AICPU |

**The AICPU never writes `claim`.** The seal lives in the acknowledgement line,
so the state graph needs no AICPU transition on the slot, which is what the
second draft's ownership table and state graph disagreed about.

**`SETTLED` is a proof, not a convention.** It is written only when
`released_after` equals `active_mask`, so the word says *every* participant of
that occupancy has left. The two phases are distinguished by their own mask
pairs: `released_before` closes the BEFORE phase (and is what lets the
participants proceed to `execute_task`), `released_after` closes the AFTER
phase, and only the latter advances the state. A participant that has not yet
set its released bit is, by definition, still counted.

#### Publication order

1. **Claim.** An arriving participant reads `claim`. If the state is `CLAIMING`
   or `READY` it is not a claimer and goes to step 4. If the state is `FREE`, or
   `SETTLED` with an occupancy other than the one it is about to create, it
   attempts `compare_exchange(claim, observed, CLAIMING | (occupancy + 1))`.
   Exactly one core wins; a loser re-reads and goes to step 4.
2. **Initialize, inside `CLAIMING`.** The winner is the only writer in this
   state, because no peer may touch any word before `READY`. It device-direct
   stores zero into both arrival masks, both released masks, both publisher
   words and `aborted`, then `OUT_OF_ORDER_STORE_BARRIER()`.
3. **Publish readiness.** The winner stores `READY | occupancy` into `claim`.
4. **Join.** A participant reads `claim` until it observes
   `READY | occupancy_it_joined`, then `fetch_or`s its bit into this phase's
   arrival mask. **This is what makes step 2 safe**: no bit exists before the
   clear, and no clear can follow a bit.
5. **Request.** The core whose `fetch_or` returned a mask that, with its own bit,
   equals `active_mask` is that phase's last arriver. It allocates the phase's
   **one immutable identity** — `phase_seq`, taken once from its own per-core
   counter — writes its request line (full `run_epoch`, full `TaskId`, phase,
   `phase_seq`, `arg_index`, `arg_cursor`, with a `request_tick` written last),
   and publishes **that** line, which is its own and has no other writer, so a
   write-back there cannot land on anyone's atomic. It then stores
   `(core_id + 1) << 32 | phase_seq` into the slot's publisher word with a single
   atomic store and touches the slot no other way.
6. **Follow.** Every peer reads that phase's publisher word and from it learns
   **which core's acknowledgement line to poll and which `phase_seq` to
   require**. A peer never published that request, so matching its own counter
   would prove nothing; it matches the publisher's. The ack line is read-only to
   peers.
7. **Progress, then seal.** The publisher re-publishes for each remaining
   argument and each remaining chunk by advancing `arg_index` / `arg_cursor` and
   the `request_tick` — **never `phase_seq`**, which is allocated once in step 5
   and is the phase's identity for its whole life. The AICPU writes
   `status`, the echoed `arg_index` / `arg_cursor` and the echoed `request_tick`
   into the publisher's ack line, then the full `(run_epoch, TaskId, phase_seq)`
   last, and flushes.

   The two readers then test different things, which is what keeps a delayed
   peer from either accepting partial progress or waiting on a stale value:

   | Reader | Accepts on | Ignores |
   | ------ | ---------- | ------- |
   | the publisher | identity match **and** `request_tick` equal to the one it last published — so it knows which request this answers | nothing; it owns the loop |
   | every peer | identity match (`run_epoch`, `TaskId`, `phase_seq`) **and** `status ∈ {Sealed, Aborted}` | `WouldBlock`, `InProgress`, `arg_index`, `arg_cursor` and `request_tick` entirely — a peer never reads progress, so there is nothing partial for it to accept and nothing per-request for it to go stale on |

   So `status` carries a *terminal* value exactly once per phase, and `phase_seq`
   never moves under a peer. The previous draft conflated the per-request
   sequence with the phase identity, which is what made a delayed peer unable to
   do either safely.
8. **Leave.** Each participant `fetch_or`s its bit into this phase's released
   mask. For BEFORE that releases it into `execute_task`; for AFTER, the core
   whose `fetch_or` completes `released_after == active_mask` stores
   `SETTLED | occupancy` into `claim` and only then may any participant publish
   the task's completion.
9. **Reuse.** The next occupant's claim (step 1) accepts `SETTLED` as a source
   state — which, by step 8, proves every previous participant has left. An
   aborted occupancy never reaches `SETTLED` and is never re-claimed in this run.

**Adversarial walk — late claimer.** A wins at step 1 and stalls inside step 2.
B reads `CLAIMING | n`: not an allowed source state, so B does not CAS, and not
`READY`, so B does not set a bit. Nothing B can do touches the line. If A never
reaches step 3, every waiter's budget expires in `CLAIMING`, where the payload
is not safely writable — so a waiter there writes **nothing**, records
`ARGS_CAPTURE_INIT_LOST` and leaves into the run's fatal path. Nothing seals and
the slot is not reused.

**Adversarial walk — stale re-claim.** C is still polling the seal for occupancy
`n` when the task completes and the slot advances. For the slot to advance to
`SETTLED | n` at step 8, `released_after` must already include C's own bit —
which only C sets, and only after it accepted the seal or the abort
acknowledgement. So a re-claim to `CLAIMING | n+1` cannot exist while C is
waiting. If C's own read sees `n+1` anyway, the only reachable cause is that C
believes it joined an occupancy it did not; C treats it as `ABORTED` for itself
and leaves into the fatal path rather than reading another occupancy's seal.

**Adversarial walk — split publishers.** BEFORE's last arriver is core 1 and
AFTER's is core 2 — the normal case, since completion order differs from
dispatch order. Each phase's publisher word is written by its own last arriver
and read by that phase's peers, so core 3 polling AFTER never reads core 1's
BEFORE sequence. A publisher word of zero means the request does not exist yet;
a seal is only ever recognised against a non-zero publisher pair, so a peer
cannot read "no request" as "sealed".

**A seal outlives its readers, not the other way round.** The acknowledgement
line holding a phase's terminal `status` may not be reused while any
participant of that phase could still read it, and the proof that none can is
the same departure proof the slot already keeps: `released_p` equal to
`active_mask`. So the publisher's ack line is free for its next phase only
after that mask closed, which for AFTER is the same instant that writes
`SETTLED`. Reuse follows departure in both places, and neither is a timeout.
Under an unknown stop nothing is reclaimed at all: every control line, every
borrowed tensor span and every arena byte the run still references is retained
by the existing stop/quarantine path, which reports rather than recovers.

### Control memory: one writer per line

Each direction gets its own line, and the one shared line is never touched by
anything but single-word atomics (above):

| Array | Count | Bytes | Writer | Reader |
| ----- | ----- | ----- | ------ | ------ |
| `ArgsCaptureRequest` | `SCHEDULER_WORKER_CAPACITY` = 108 | 108 × 64 = 6.75 KiB | the core | AICPU |
| `ArgsCaptureAck` | 108 | 6.75 KiB | AICPU | the publishing core **and its phase peers** |
| `ArgsCaptureSlot` | `SCHEDULER_CLUSTER_CAPACITY`(36) × `SCHEDULER_PENDING_SLOT_COUNT`(2) = 72 | 72 × 64 = 4.5 KiB | per field, per the ownership table above | the arrivers |
| AICPU per-core request state | 108 entries | ~1.7 KiB of AICPU memory | AICPU | AICPU |
| AICPU open reservation | **one per dump lane**; a lane is an AICPU thread's, so `PLATFORM_MAX_AICPU_THREADS` = 5 | 5 × 64 = **320 B** of AICPU memory | AICPU | AICPU |

**Total GM: 18 KiB**, reserved only when the bind enables ArgsDump — not the
6.75 KiB the first draft promised, which counted the request array alone — plus
**~2.2 KiB of AICPU memory**. The reservation table is 320 B rather than the
6.75 KiB of the third draft because a lane holds **one** open reservation, which
is the subject of the arena section below.

The arrival and released masks are the fields with more than one writer; each
participant touches its own bit through the scheduler's existing GM fetch-or,
and no reader of those words is outside the participant set.

### Ordering, and stale acknowledgements

- Request: the core writes `task_id`, `phase`, `phase_seq`, `arg_index`,
  `arg_cursor`, then `request_tick` last, with `OUT_OF_ORDER_STORE_BARRIER()`
  before that tick and `dcci(line, CACHELINE_OUT)` + `dsb` after it — the shape
  `pmu_aicore_record_task` already uses. The AICPU
  `cache_invalidate_range(line, 64)` before every read.
- Acknowledgement: the AICPU writes `status`, `arg_index`, `arg_cursor` and the
  echoed `request_tick`, then `(run_epoch, TaskId, phase_seq)` last,
  `cache_flush_range` after. The core `scheduler_observe_cache_line(ack)`
  before each read.
- **Two identities, and only one of them is the phase's.** `phase_seq` is
  allocated once per `(occupancy, phase)` by that phase's last arriver and
  never changes; `request_tick` is per retry and per chunk and changes
  constantly. `run_epoch` and `TaskId` are carried whole. A **peer** accepts an
  acknowledgement only on `(run_epoch, TaskId, phase_seq)` plus a terminal
  `status`, so an ack left by a previous run, a previous phase or a previous
  occupancy of the same line is ignored, and no progress reply can be read as a
  seal. The **publisher** additionally matches `request_tick`, which is how it
  knows which of its own requests was answered. Nothing is matched against a
  truncated identity anywhere.
- **Payload visibility.** `pipe_barrier(PIPE_ALL)` orders the core's own pipes,
  not the write-back the AICPU then reads, so before publishing an AFTER
  request the core also issues `dcci(CACHELINE_OUT)` over the output spans it
  wrote; the AICPU invalidates the tensor extent before copying. Treating the
  pipe barrier alone as publication is the a2a3 completion-signal trap.

### Bounded service, and the producer surface it needs

**`dump_arg_record` cannot be a service quantum, and one argument is not a
bound.** In `args_dump_aicpu.cpp` the call is all-or-nothing and blocks in two
places, and the work it does is unbounded in a third:

| In the call | What it does | Why it is not a quantum |
| ----------- | ------------ | ----------------------- |
| `ensure_dump_arena_capacity` | spins `while (state->completed_payload_count < target_payload_count)` until the host's acknowledgement catches up, bounded only by `kDumpQueueBackpressureWaitCycles` | the supervisor is inside the spin and polls no core, no error latch and no retirement |
| `switch_dump_meta_buffer` → `DumpEngine::switch_buffer` | publishes the full metadata buffer and waits on the ready queue's own back-pressure budget | same, on the metadata side |
| `write_dump_arg_logical_prefix` | copies `copy_bytes`, which is clamped to `arena_size / 2` — **64 MiB** — for a tensor larger than the arena, and which `gather_dump_arg_dim` emits per innermost row, or per *element* when the innermost stride is not 1 | one argument can be the whole service budget |

So the mechanism needs a resumable form of the producer, and that is part of
the contract delta rather than a free call:

```text
dump_arg_try_begin(thread_idx, info, &reservation)
    -> Accepted | WouldBlock | Refused(reason)
       refuses while this lane already holds an open reservation, else
       acquires the metadata buffer and tests the arena predicate WITHOUT
       spinning, reserves copy_bytes once, and returns the reservation:
       { meta buffer, record index == count, payload offset, copy_bytes,
         pre-reservation arena offset, element cursor, bytes_written }
dump_arg_copy_chunk(&reservation, max_bytes) -> bytes_copied
       resumes the (possibly strided) gather from the cursor; the existing
       CircularArenaWriter already carries bytes_written, so the cursor is
       that plus the element index the gather reached
dump_arg_commit(&reservation)
       writes the metadata record, advances count by one, counts it, and
       closes the lane's reservation
dump_arg_abandon(&reservation)
       rolls arena_write_offset back to the reservation's pre-reservation
       value, leaves count where it was, accounts exactly one refused record
       and closes the lane's reservation
```

A service step is then **one chunk of `max_bytes`**, not one argument: the
supervisor copies at most that many bytes, writes the advanced cursor into the
acknowledgement, and returns to its loop, so error polling, retirement and the
other cores' requests get a turn inside a large or strided argument. The
publisher re-publishes with the advanced cursor until the phase is sealed.

**Would-block is not a refusal, and only one of them is accounted.** The
existing call collapses both into `-1` plus `account_refused_record`, which is
exactly what makes a retry double-count:

| Outcome | Cause | Accounting | Retry |
| ------- | ----- | ---------- | ----- |
| `WouldBlock` | arena predicate not yet satisfied, no free metadata buffer yet, **or this lane's one reservation is open for another core** | **none** | yes — re-tested on the next service step, in FIFO arrival order |
| `Refused(Permanent)` | no header, lane off, invalid thread, level `OFF` | one refused record, once | no |
| watchdog | a `WouldBlock` that never cleared within `scheduler_timeout_cycles` | **only if a reservation is open on this lane**, one `dump_arg_abandon` — one refused record, once. A would-block that never opened a reservation has nothing to abandon and is **not** charged a refusal: it was never a record. The no-reservation abort walk below is authoritative | no |

Nothing becomes visible to the host before `dump_arg_commit`, so a retry before
commit duplicates no record.

#### One open reservation per lane, and why

A resumable wrapper is not by itself proof that the lane is sound, because the
existing `DumpMetaBuffer::count` is **both** the allocation cursor and the
publication watermark, and the arena's `arena_write_offset` is both too. With
two reservations open on one lane the schedule that breaks it is immediate:
A reserves record 0 and yields mid-copy; B reserves record 1 and commits;
advancing `count` to 2 publishes A's record 0, which holds nothing.

**So a lane holds at most one open reservation.** Not a simplification of the
accounting — a property that makes the two cursors coincide again:

| Invariant | Consequence |
| --------- | ----------- |
| `reserved == count + (open ? 1 : 0)` | the open record is always the one at `count`, so `commit` advancing `count` by one can never publish an uninitialized record and no second watermark is needed |
| the open reservation's arena bytes are the last bytes taken | `abandon` **rolls `arena_write_offset` back** to its pre-reservation value; an abandoned reservation leaks no arena capacity at all, which the previous draft conceded it would |
| a metadata buffer is published only while no reservation is open | a buffer switch waits for the commit rather than racing it, so a published buffer's every record is complete |
| the host-facing counters move only at commit | `total == collected + dropped` holds across any number of retries |

**Service and fairness — what ordering actually exists.** Independent cores
publish their requests without a shared ticket, so **true publication order is
not recoverable** by any wording: two requests that appear between one scan and
the next are unordered as far as this lane can ever know. What the supervisor
does have is its own **observation order**, and the guarantee is stated in those
terms:

- The supervisor scans every core of its lane once per service step, so a
  published request is observed within one scan of being published.
- On first observation a request is stamped with the lane's next `observe_tick`
  (the supervisor's own counter, not shared with any core) and queued.
- Reservations open in `observe_tick` order, and a queued request keeps its tick
  across any number of `WouldBlock` answers.

The property that follows is the one fairness needs: **a core is overtaken at
most once by a peer that published later**, because after the scan in which both
are visible their relative order is fixed forever. So the wait is bounded by the
number of cores on the lane times one reservation, not by a lottery. What is
*not* claimed is that the earlier publisher is always served first — within a
single scan the order is the scan's, and the design does not pretend otherwise.
Encoding real publication order would need a shared lane ticket the cores
`fetch_add`, which is one more GM word and one more atomic on the request path;
it is not taken, because the bound above is what fairness requires and the extra
word is not free.

**Back-pressure.** Two sources, both reported the same way and neither blocking
the supervisor: the arena predicate (the host has not acked enough payloads yet)
and the lane being busy with another core's reservation. Both are `WouldBlock`;
both are bounded by the waiter's own watchdog, after which the run fails with
its accounting unknown rather than publishing a partial record.

**What it costs.** Head-of-line latency: a 64 MiB clamped argument delays the
next core's capture by its own copy. That cost lands on a core which is *already*
parked at its barrier waiting for its own seal, and the lane still makes
progress, so it lengthens the capture rather than stalling the run. The storage
drops with it — the reservation table is 5 × 64 B rather than 108 × 64 B, and
the cost table above is updated accordingly.

**Adversarial walk — the A/B schedule.** A reserves record 0 and yields mid-copy.
B's request is observed; the lane is busy, so B is answered `WouldBlock` and
nothing is reserved or accounted for B. A resumes a chunk per service step and
commits; `count` advances 0 → 1. B's next retry opens its reservation at
record 1. No buffer was published in between, because a switch is refused while a
reservation is open.

**Worked walk — two arguments, several chunks, a would-block, an abort after a
commit.** Core A is the AFTER publisher for a 1C2V task; core B and core C are
its peers; the phase carries two tensors, the second of them arena-clamped.

| Step | Publisher A | AICPU on lane L | Peers B and C |
| ---- | ----------- | --------------- | ------------- |
| 1 | publishes `phase_seq = 9`, `arg_index = 0`, `cursor = 0`, `tick = 1`; stores `publisher_after = (A+1)<<32 \| 9` | observes the request, stamps `observe_tick`, opens a reservation for argument 0 | read `publisher_after`, learn to poll A's ack for `phase_seq = 9` |
| 2 | — | copies a chunk; acks `status = InProgress`, `arg_index = 0`, `cursor = 64 KiB`, `tick = 1` | ignore it: `status` is not terminal |
| 3 | republishes `cursor = 64 KiB`, `tick = 2` | copies the rest; `commit` writes record `count`, advances `count`, closes the reservation; acks `InProgress`, `arg_index = 0`, `cursor = done`, `tick = 2` | still waiting |
| 4 | republishes `arg_index = 1`, `cursor = 0`, `tick = 3` | another core's reservation is open on L → acks `status = WouldBlock`, `tick = 3`. **Nothing accounted** | still waiting; `WouldBlock` is not terminal |
| 5 | republishes unchanged, `tick = 4` | reservation now free; opens one for argument 1 (the clamped tensor), copies chunk 1 of 1024 | still waiting |
| 6 | A's watchdog expires mid-argument-1; `compare_exchange(aborted, 0 → 1 \| site)`, records `ARGS_CAPTURE_TIMEOUT`, **keeps polling** | observes `aborted`: finishes the chunk in flight, `dump_arg_abandon` — arena offset rolled back to the pre-reservation value, `count` left where argument 0's commit put it, **one** refused record accounted — then acks `status = Aborted` with `(run_epoch, TaskId, 9)` | — |
| 7 | reads `Aborted` for `phase_seq = 9`, leaves into the run's fatal path | — | read the same terminal `status` for the same `phase_seq`, leave the same way |

What the host sees: argument 0's record is committed and complete, argument 1 is
one refused record, nothing was sealed, `released_after` is never completed so
the slot stays at `READY | n` and is not re-claimed, and the lane's terminal is
never closed — so the run's argument accounting reports unknown. **The committed
prior argument is not retracted**: it was a complete record of a real argument at
a real stage, and `total == collected + dropped` still holds with the abandon
counted once.

**Abort with no reservation open.** The same step 6 with A's watchdog expiring
in step 4's `WouldBlock`, before argument 1 was ever reserved. There is nothing
to abandon: `dump_arg_abandon` is not called, no refused record is accounted
(the would-block never was one), the arena offset and `count` are already exactly
what argument 0's commit left, and the AICPU acks `Aborted` from the same place.
The only difference between the two aborts is whether one refused record is
charged, and that follows from whether a reservation existed — never from the
abort itself.

**Delayed peer.** Core C is descheduled for the whole of steps 1–6 and wakes at
step 7. It reads `publisher_after` once, polls A's ack, and the first thing it
ever sees is a terminal `Aborted` carrying `phase_seq = 9` — the same word every
peer saw, because `phase_seq` never moved and the progress fields it never reads
are the only things that did. Had it woken at step 3 instead, it would have seen
`InProgress` for `phase_seq = 9` and kept waiting: not partial acceptance, and
not a stale sequence to poll forever on.

**Adversarial walk — abandon under a full buffer.** A holds record 255 of a
256-record buffer and times out. `abandon` rolls the arena offset back, leaves
`count` at 255 and accounts one refused record. The buffer is now publishable
(no open reservation), the switch proceeds, and B opens record 0 of the next
buffer. Nothing is wedged and no record names bytes that were never written.

Arena pressure is therefore the existing handshake, with the wait moved out of
the producer and into the supervisor's own loop. **Worst case is priced, not
capped**: `FULL` on the chained acceptance workload is 513 tasks × up to 3
tensors × 64 KiB ≈ 96 MiB against a 128 MiB arena, so a larger graph *will*
back-pressure — which is why the mechanism has to carry flow control rather
than redefine the caller's mode. `PARTIAL`, `HYBRID` and `FULL` keep exactly
their current selection semantics; the earlier suggestion to cap `FULL` to the
marked set is withdrawn.

### Failure, abort and close — one stop transition

The first draft had three different protocols here: cores leaving immediately on
`aborted`, the supervisor settling before any release, and an aborted slot being
re-claimed by the next generation. **One is chosen: abort waits for the
supervisor's acknowledged settlement, and an aborted generation ends the run.**
Nothing leaves on the flag alone.

| Step | Actor | Action |
| ---- | ----- | ------ |
| 1 | the first waiter whose budget expires in `READY` | `compare_exchange(aborted, 0 → 1 \| site)` — the atomic is the publication, and **no line operation follows it**; the waiter records `ARGS_CAPTURE_TIMEOUT` as the run's error and **keeps polling** the ack line |
| 2 | the AICPU, on observing `aborted` for a slot it is serving | finishes or drops the **current chunk** — never abandoning mid-copy — then calls `dump_arg_abandon` **only if a reservation is actually open on this lane**, and writes `status = Aborted` with `(run_epoch, TaskId, phase_seq)` into the publisher's ack line and flushes. No slot word is written and no line is published |
| 3 | every participant | leaves only on the seal **or** on `status == Aborted` carrying the full identity this phase published; it never fabricates either |
| 4 | every participant, after release | does not execute the task, publishes no completion, takes no further task — it goes to the run's shutdown path |
| 5 | the slot | `released_after` is therefore never completed by a release that was not acknowledged, so `claim` stays at `READY \| occupancy`: **this occupancy is never re-claimed in this run**, and the next run re-initializes the array |
| 6 | release of memory | no control line, tensor span or arena byte is released until the producer stop (`retired_core_count_`) **and** the lane holds no open reservation |

A waiter whose budget expires while the slot is still `CLAIMING` writes nothing
at all (the payload is not safely writable there), records
`ARGS_CAPTURE_INIT_LOST`, and leaves into the same fatal path. If the AICPU
never acknowledges at all, a second, longer budget expires into the run's fatal
path with nothing sealed.

**Retirement is not the only overwrite risk, and the barrier is what closes the
other one.** Borrowed allocation lifetime says the bytes still exist; it says
nothing about a *successor task in the same run* writing them while a chunk is
being copied. That window is closed by the AFTER barrier itself: the next task
cannot be dispatched until this task publishes completion, and completion is
published only after the AFTER phase is sealed or acknowledged as aborted. The
BEFORE phase is covered in the other direction — this task was dispatched only
because its predecessor already completed. So the abort path must *also* wait
for the acknowledgement, which is exactly why step 3 does not release on the
flag; a release on the flag alone would reopen the window the barrier exists to
close.

The remaining lifetimes are unchanged: the original tensors are the caller's,
borrowed by the lane for the whole run (`borrow_device_spans`); the host payload
is released by the existing per-payload acknowledgement; the dump lane's
terminal is never closed on an aborted run, so the host reports its argument
accounting unknown and quarantines — the behaviour that already covers an
unsettled producer. Diagnostics off: nothing is reserved, no request is
published, no round-trip is paid.

### One transition table

Both phases, the slot gate, the lane reservation and the abort path in one
table. `S` is the slot, `n` its occupancy, `M` the task's `active_mask`, `p` the
phase (BEFORE or AFTER), and `L` the dump lane serving `S`'s cluster. Every
slot access named here is a single 64-bit atomic or device-direct store; no row
writes back or invalidates the line.

| # | From | Event | Actor | To | Side effects and what it proves |
| - | ---- | ----- | ----- | -- | ------------------------------- |
| 1 | `claim = FREE`, or `SETTLED\|m` with `m ≠ n` | `compare_exchange(claim, observed, CLAIMING\|n)` succeeds | one arriving participant | `CLAIMING\|n` | The source state proves no participant of any occupancy is inside: `SETTLED` was written only when `released_after == M` (row 10). `CLAIMING`/`READY` are not source states, so no late core can re-enter and re-clear. |
| 2 | `CLAIMING\|n` | zeroes both arrival masks, both released masks, both publisher words, `aborted`; `OUT_OF_ORDER_STORE_BARRIER()` | the claimer, sole writer in this state | `CLAIMING\|n` | No peer may touch any word before `READY`, so the clear races nothing. |
| 3 | `CLAIMING\|n` | `store(claim, READY\|n)` | the claimer | `READY\|n` | Initialization is complete before any bit can exist. |
| 4 | `CLAIMING\|n` | waiter's budget expires | any participant | **run fatal** | Writes nothing — the payload is not safely writable here. `ARGS_CAPTURE_INIT_LOST`; no seal, no reuse. |
| 5 | `READY\|n` | `fetch_or(arrived_p, my_bit)`, result ∪ my_bit ≠ `M` | each participant | `READY\|n` | One writer per bit; the core waits. |
| 6 | `READY\|n` | `fetch_or(arrived_p, my_bit)`, result ∪ my_bit = `M` | the **last arriver of `p`** | `READY\|n`, `p` requested | Allocates this phase's one immutable `phase_seq`, writes its own request line (full `run_epoch`, full `TaskId`, `p`, `phase_seq`, `arg_index`, `arg_cursor`, `request_tick` last), publishes that line, then `store(publisher_p, (core+1)<<32 \| phase_seq)`. Exactly one publisher per `(n, p)`, decided by the primitive; `phase_seq` never moves again, and every later retry or chunk changes only `request_tick`. |
| 7 | `p` requested | `try_begin` → `WouldBlock` (arena predicate, no free buffer, or `L`'s reservation open for another core) | the AICPU | `p` requested | Nothing reserved, nothing accounted; the publisher republishes with a new `request_tick` and the same `phase_seq`. Observation order bounds the retry: a core is overtaken at most once by a later publisher. Peers ignore this status entirely. |
| 8 | `p` requested | `copy_chunk(max_bytes)`, arguments remain | the AICPU | `p` requested | `status = InProgress` with the echoed `arg_index` / `arg_cursor` / `request_tick`; the supervisor returns to its loop between chunks. Only the publisher reads these fields — a peer tests `status` for a terminal value and nothing else, so no peer can accept partial progress. |
| 9 | `p` requested | last argument committed; ack carries `status = Sealed`, the full identity and `phase_seq` | the AICPU | `p` sealed | `count` advances one per commit, and the open record is always at `count`, so no uninitialized record is published. |
| 10 | `p` sealed | `fetch_or(released_p, my_bit)` | each participant, after matching the full identity | `p` released | BEFORE: the core proceeds to `execute_task`. AFTER: the core whose result ∪ my_bit = `M` stores `SETTLED\|n` — the proof that **every** participant left — and only then may completion be published. |
| 11 | `READY\|n` or `p` requested | `compare_exchange(aborted, 0 → 1\|site)` | the first waiter to time out | same state, `aborted` set | The waiter keeps polling; it does not leave on its own flag. |
| 12 | `aborted` set | settles the chunk in flight; **if a reservation is open** `dump_arg_abandon` (arena offset rolled back, `count` unchanged, one refused record), **if none is open** nothing is abandoned and nothing is accounted; then `store(ack.status, Aborted)` with the full identity and `phase_seq`. No slot word is written and no line is published. | the AICPU | **run fatal** | Participants leave only on this acknowledgement (row 3 of the abort table above), which is what keeps a successor from overwriting bytes still being copied. A committed prior argument stays committed. |
| 13 | `SETTLED\|n` | row 1 by the next occupant | the next task's first arriver | `CLAIMING\|n+1` | Reachable only after row 10 completed `released_after`, so reuse strictly follows every departure. |

**Diagnostics off** short-circuits before row 1: no slot is claimed, no request
published, no reservation opened.

A participant that reads a `claim` whose occupancy is not the one it joined
stops polling and leaves into the run's fatal path; it never reads another
occupancy's seal.

## Mechanism S — AICore-side staging, and why it is not cheaper

Staging the record and copying the payload on the core removes the wait only by
moving the arena's *producer* from the AICPU to the AICore: the free/ready queue
protocol, the per-lane payload cursor and the truncation accounting are
AICPU-owned, and the AICore cannot reach the host allocator. It also charges a
64 KiB-per-tensor copy to the task it is measuring. Staging only a *descriptor*
and copying later does not work at all — the bytes it names are overwritten by
the next task, which is the defect this entry exists for.

## Recommendation

**Mechanism R as specified above, with `PARTIAL`/`HYBRID`/`FULL` semantics
unchanged.** It would close the gap the refusal currently names: every captured
record is the content its own task read or wrote, mix tasks produce one
correctly-attributed record set, and loss is reported rather than invented. The
resumable producer surface is the part that has to land first — without it the
supervisor's service step is a blocking call, and no barrier design on top of
that is bounded.

### The contract delta a future change would take

Deferred, not pending: a5 `host_build_graph` ships as a runtime without this
capability, and the rows below are what adding it would cost. Nothing in the
tree waits on them.

| Change | Kind | Cost |
| ------ | ---- | ---- |
| `ArgsCaptureRequest` / `ArgsCaptureAck` / `ArgsCaptureSlot` arrays in the scheduler state, reserved only when the bind enables ArgsDump | new resource | **18 KiB GM** per run, zero when the owner is off, plus **~2.2 KiB of AICPU memory** (per-core request state and one open reservation per lane) |
| A resumable producer surface on the dump lane — `dump_arg_try_begin` / `dump_arg_copy_chunk` / `dump_arg_commit` / `dump_arg_abandon`, with the existing blocking `dump_arg_record` kept for the AICPU-dispatch routes, and **one open reservation per lane** served FIFO | **new producer API** | no new *shared* storage and no second watermark: the open record is always the one at `count`, an abandon rolls the arena offset back, and a buffer is never published with a reservation open. The arena and metadata waits move from inside the producer to the caller's loop; the refusal counter gains a would-block outcome that is not accounted. Head-of-line latency is the cost — one core's clamped argument delays the next core's capture, on cores already parked at their own barrier |
| Participating cores block at a task's two boundaries until the AICPU seals the phase or acknowledges an abort | **new ordering contract** | 2 barriers + ≥1 GM round-trip per captured argument, plus one more per `max_bytes` chunk of a large or strided one, on the dispatch path, bounded by the existing scheduler watchdog; nothing for an unmarked task under `PARTIAL` |
| `ARGS_CAPTURE_TIMEOUT` / `ARGS_CAPTURE_INIT_LOST` error sites, and an abort that ends the run rather than releasing a slot for reuse | new failure mode | a capture timeout fails the run and leaves its argument accounting unknown; no slot of that run is re-claimed |

Nothing else in the diagnostics contract would move: no second budget, no new
verdict, no change to admission or to the stop-proof model, and no change to
what `PARTIAL`/`HYBRID`/`FULL` select.

## Acceptance

1. **Mixed subtasks** — a 1C2V task yields one record set carrying all three
   `func_ids`, and no subtask's `execute_task` begins before its input capture
   is sealed.
2. **Concurrent cores** — every cluster requesting at once completes with no
   core starved and no slot left `READY` or `REQUESTED`.
3. **Claim race** — with every participant attempting the claim at once,
   exactly one initializes, no arrival bit is lost, a core arriving while the
   slot reads `CLAIMING` or `READY` never becomes a second claimer, and a core
   descheduled in `CLAIMING` ends the run rather than sealing a
   half-initialized slot.
4. **Split publishers** — a phase whose BEFORE and AFTER last arrivers are
   different cores: each peer polls the ack line the slot's own publisher word
   names, and no peer accepts the other phase's sequence.
5. **Departure proof** — a slot advances to `SETTLED` only when every
   participant's released bit is set, and the next occupancy's claim is
   observed never to precede the last departure, including when one
   participant's seal arrives far later than its peers'.
6. **Lane reservation** — the A/B schedule: with one core holding a reservation
   mid-copy, a peer's request is answered `WouldBlock` with nothing accounted,
   no metadata buffer is published while that reservation is open, and after an
   abandon the arena offset and `count` are exactly what they were before the
   reservation opened.
7. **Exact handoff** — the four-backend scenes require the first marked task's
   output argument to read `a + b` at its own completion, the second marked
   task's input argument to read `a + b` at its own dispatch and its output
   `a + 2b`, each matched on run, task, argument index and stage, with the
   execution order taken from the swimlane rather than the bytes. Those pass on
   this route only when the capture is at the task boundary.
8. **Arena pressure** — `FULL` on the chained workload back-pressures without
   deadlocking; records complete or the run reports unknown, never short. A
   would-block that later clears adds **no** refused record, and the host's
   `total == collected + dropped` still holds after a phase that was retried.
9. **Bounded service** — with one argument sized at the arena clamp
   (`arena_size / 2`) and one strided with a non-unit innermost stride, the
   supervisor still polls every other core, the error latch and retirement
   between chunks, and no service step exceeds `max_bytes`. Queue order is
   **first observation**, which is the only order this lane can establish, and
   a reservation already open keeps advancing rather than yielding to a request
   waiting for the same space. **No wall-clock bound is claimed from a finite
   queue**: progress depends on the host acknowledging payloads, and a lane
   that stops progressing is covered by the watchdog and the stop path, not by
   the queue being short.
10. **Failure/close** — an induced capture timeout leaves the lane unsettled and
    the bank quarantined, with no sealed partial phase, no record committed from
    the abandoned reservation, no participant released before the acknowledgement
    and no tensor or control line released before the producer stop.
11. **Diagnostics off** — no control storage reserved, no request published,
    no rendezvous entered. Not a claim of zero instructions: the selection test
    itself remains on the path, as it is today.
