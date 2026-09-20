# 2026-09 — Porting the a2a3 AICore retirement protocol to a5: what was dropped on the way

**Verdict: ported, minus two mechanisms a5 has no counterpart for, and with one
deliberate difference.** Shipped as #2388 (issue #2387). This entry records the
alternatives that were measured and dropped, so they are not re-proposed.

Supersedes the verdict of the entry proposed in #2300 ("dropped, no net gain"),
which was measured over a shape this one does not use. See *Where the earlier
verdict went wrong* below.

## Setting

a5 retired AICores one at a time: write EXIT, block for that core's ACK
against a per-core deadline, write IDLE. a2a3 retires the same hardware as a
claimed group with a read-back close. The question was which of a2a3's
mechanisms apply to a5, and what they cost.

Measurements: a5 development card, nine graphs by three blocks, paired within
each block, medians of 20 timed calls per process, five rounds of 171
processes. The segment is the retirement tail — `graph_build` end minus
`max(orch end, sched end)` — derived from the `[STRACE]` markers already
emitted. Parenthesised counts are blocks where the sign held.

## What shipped

Eight mechanisms: the claim, `wmb` after the broadcast, a non-blocking
round-robin sweep against one shared deadline, deferred window close, the
close read-back, the `rmb` that drains it, per-core reporting of cores not
released, and fatal-before-completion publication. Net **-1.43 us (0/12)**
against a5 before the change, about 1% of a call.

## Dropped: fast-path window control, and the GM return gate

Not a cost decision. a5 has no fast-path enable anywhere in its software — no
offset, no `RegId` member, no `reg_offset` case, no AICore-side case, no
caller. Writing the dispatch register to idle is what opens a window there, and
there is no separate state to close. The return gate's condition, in a2a3's
own comment, is that the AICPU opens it "only after it has closed this core's
fast-path window"; with no such window the gate guards nothing, and
`src/common/task_interface/aicore_teardown.h` already records that a5 leaves it
unused.

**These are one item, not two.** Without the first the second has no meaning.
Re-open only if a5 gains a window control with state — and note that whether
the *silicon* carries an equivalent is not something this repository can
answer. What is established is that a5 neither opens nor closes one.

a5 still needs the ordering the gate contributes to on a2a3 — a retired core
not touched again before its close has landed. That comes from the read-back
plus dispatch having stopped before retirement begins, rather than from a
per-core mechanism. Worth revisiting if cores ever become reusable within a run.

## Dropped: three ways to make the claim cheaper

The claim is the whole cost. Taken as a2a3 takes it, per core, the port
measures **-12.35 us (0/27)**; everything else together measures
**-0.40 us (5/27)**, i.e. free — the sweep's batched broadcast pays for the
read-back. So three attempts went at the claim itself:

| Attempt | Result |
| ------- | ------ |
| Move the flags onto each core's own cache line (`CoreExecState` is already `alignas(64)` and the retiring thread has just read `reg_addr` from that line) | **Recovers half.** -12.35 to -7.03. The remaining cost is not false sharing |
| Relax the claim to `__ATOMIC_RELAXED` (sound: the loser only skips the core, and the winner's `reg_addr` was published at handshake) | **Nothing.** -8.56, no better than `acq_rel` |
| Take one claim per owning scheduler thread instead of one per core | **Works.** 8.24 us to 0.98 us |

The cost is thread scatter, not work, and it is proportional to how many times
the claim is taken — not to where the flags live or how they are ordered. A
probe arm with no claim at all isolates it: 8.24 us (27/27) between that arm
and the same build with a per-core claim, and the claim is the only thing that
inflates the broadcast phase, which contains no claim at all.

Per-thread is also the right granularity on the merits:
`assign_cores_to_threads` hands every cluster to exactly one scheduler thread,
so a thread's core set is the smallest unit the normal and the emergency path
can contend for. a2a3 claims per core because its emergency path walks cores.

**Do not re-propose cache-line placement or memory ordering for this claim.**

## Dropped: two micro-optimizations of the close

| Attempt | Result |
| ------- | ------ |
| Split the close into a store burst then a read-back burst, so the read-backs pipeline instead of each draining its own store | **No distinguishable difference** (+0.05, 7/12). The predicted ~2 us from the STR/LDR round-trip cost table did not appear; that is not the bottleneck in this segment |
| Close each window on its ACK instead of deferring every close until the group has acknowledged | **No distinguishable difference** (+0.68, 21/27, below the segment's 1 us floor). a2a3 defers because its close also releases the return gate — a reason that does not apply to a5 — but removing the deferral buys nothing, so the shape stays as a2a3 writes it |

## Ruled out on architecture, not measurement

- **A `dsb` instead of the read-back.** The window is Device-nGnRE; the E is
  the early write-acknowledgement, which is what the barrier waits for.
- **One read-back covering the group.** nR orders accesses within one
  peripheral, minimum 4 KB granule; each core has its own window.
- **Reading back `COND` rather than `DATA_MAIN_BASE`.** Same reason —
  `DATA_MAIN_BASE` (0xD0) and `COND` (0x5108) are not in the same granule, so a
  COND load does not order against the close.
- **A shared deadline with a blocking per-core wait** (what #2288 proposed).
  Exit waiting has two independent properties: every core judged on its own
  full budget, and the group bounded by one budget. The per-core wait has the
  first, a shared deadline with a blocking wait has the second, and only the
  non-blocking sweep has both.

## Where the earlier verdict went wrong

PR #2300 proposed recording this as "dropped, no net gain". Two of its findings
hold and are carried forward: the two non-portable mechanisms, and the
diagnostic round's eliminations (`runtime_destroy` unchanged, no extra COND
reads from the sweep). The verdict does not, for three reasons.

1. **"Only three steps have an a5 counterpart" undercounts.** That was measured
   over the wait-and-close sequence alone. The claim, the read-back, its drain,
   fatal-before-completion, and per-core reporting also port: eight of ten, not
   three.
2. **The read-back was treated as tied to fast-path close and dropped with it.**
   It is independent. a5's own close is a posted store to a Device-nGnRE window,
   and the run publishes completion through a Normal-cacheable release store
   that cannot order it.
3. **Its 59% transfer figure is specific to the per-core claim.** Per thread the
   whole port lands at -1.43 us.

## Still open

Fault injection. None of these mechanisms can be validated by a normal
benchmark — their value is entirely on the fault path, where a benchmark can
only price them. Coverage needed: slow and unresponsive cores against both
budget properties, concurrent normal and emergency retirement, and the
fatal-before-completion ordering. Tracked on #2388.
