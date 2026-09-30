# 2026-09 — Porting the a2a3 AICore retirement protocol to a5: what was dropped on the way

**Verdict: preserve a2a3's per-core claim, emergency batching and post-close
return gate.** PR #2388 (issue #2387) ports these to a5 TMR and covers the
corresponding HBG teardown paths. Separate fast-path
register control has no defined a5 software counterpart. Historical measurements
below describe earlier candidates, not the current per-core gated implementation.

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

## Protocol

The mechanisms are the per-core claim, `wmb` after the broadcast, a non-blocking
round-robin sweep against one shared deadline, deferred window close, the
close read-back, the `rmb` that drains it, per-core reporting of cores not
released, fatal-before-completion publication, and a GM return gate released only
after close and its drain. Normal owners retire in parallel. Emergency retirement
collects all ready cores it wins into one broadcast and one deadline.

## Platform adaptation: fast-path window control

Not a cost decision. a5 has no fast-path enable anywhere in its software — no
offset, no `RegId` member, no `reg_offset` case, no AICore-side case, no
caller. Writing the dispatch register to idle is what opens a window there, and
there is no separately defined FAST_PATH close operation. This establishes the
software interface, not the absence of an equivalent on silicon; copying a2a3's
register offset without an a5 definition would be unjustified.

The earlier conclusion that this also made the GM return gate unnecessary was
too strong. The gate independently orders the worker's return after the AICPU's
last IDLE write, read-back and drain. TMR therefore uses isolated per-core gates,
reset before window-open. The worker acknowledges EXIT, then waits on its gate
before returning. Unacknowledged cores remain unreleased for host recovery.
The HBG resident and legacy paths use the same post-close gate rule. The
resident path keeps its cross-thread EXIT broadcast before each partition's
ACK sweep; the legacy path claims cores so normal and emergency retirement
cannot close or release the same worker twice.

## Historical claim-cost experiments

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

Disjoint ownership proves only that a per-owner claim can prevent duplicate
retirement. It does not prove equivalent emergency progress. The per-owner
implementation called the blocking platform helper once per owner, with a fresh
deadline each time. If the emergency caller alone handles several ready groups
with silent members, later groups receive EXIT only after earlier timeouts;
the wait can approach one budget per group. a2a3 broadcasts its entire claimed
emergency set first and spends one budget on that set. The port restores that
behavior and per-core arbitration, including overlapping target subsets.

The historical 8.24/0.98 us comparison does not price the current initialization
handoff or return gate, and is not a reason to weaken the fault protocol.

**Do not re-propose cache-line placement or memory ordering for this claim.**

## Dropped: two micro-optimizations of the close

| Attempt | Result |
| ------- | ------ |
| Split the close into a store burst then a read-back burst, so the read-backs pipeline instead of each draining its own store | **No distinguishable difference** (+0.05, 7/12). The predicted ~2 us from the STR/LDR round-trip cost table did not appear; that is not the bottleneck in this segment |
| Close each window on its ACK instead of deferring every close until the group has acknowledged | **No distinguishable difference** (+0.68, 21/27, below the segment's 1 us floor). The measured candidate kept the deferred close; the current A5 port also releases each acknowledged core's return gate only after the close/read-back/drain sequence |

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
hold and are carried forward: the absence of a separately defined A5 FAST_PATH
register operation, and the
diagnostic round's eliminations (`runtime_destroy` unchanged, no extra COND
reads from the sweep). The verdict does not, for three reasons.

1. **"Only three steps have an a5 counterpart" undercounts.** That was measured
   over the wait-and-close sequence alone. The claim, the read-back, its drain,
   fatal-before-completion, and per-core reporting also port. The historical
   candidate counted eight of ten, rather than three; the current port also
   carries the independent GM return gate.
2. **The read-back was treated as tied to fast-path close and dropped with it.**
   It is independent. a5's own close is a posted store to a Device-nGnRE window,
   and the run publishes completion through a Normal-cacheable release store
   that cannot order it.
3. **Its 59% transfer figure is specific to the per-core claim.** The earlier
   per-thread candidate measured -1.43 us for the whole port; this does not
   measure the current per-core claim, initialization handoff and return gate.

## Fault injection: where it had to move to

None of these mechanisms can be validated by a normal benchmark — their value
is entirely on the fault path, where a benchmark can only price them. The
first attempt drove the fault path from the host system tests and established
nothing: an unresponsive core forces a device reset, so the log line naming
the core never reaches the host, anything timed around the retirement is
dominated by the reset instead, and a test that counts retirement log lines
passes vacuously when no lines arrive at all. All three were withdrawn.

What replaced them is a device-side unit suite driving
`platform_retire_aicore_group` against simulated register blocks, in the style
the a2a3 suite already uses
(`tests/ut/cpp/a5/platform/test_aicore_retirement.cpp`).
Seven cases cover: the broadcast completing before any core is waited on; no
window closing until the group has acknowledged; a silent core left unclosed
while its answering peers are released; the `released[]` contract; one shared
budget for the group rather than one per core; address validation; and the
single-target group inheriting all of it. Three more cover the return
gates: release only after close and only to acknowledged cores, a gate held shut
while a peer's ACK is pending, and an invalid target rejecting the whole batch
before any EXIT is signalled. That is ten platform cases in total.

`tests/ut/cpp/a5/platform/test_return_gate_wait.cpp` adds three more: they include
the real a5sim `aicore/inner_kernel.h` and run the actual inline
`wait_for_post_close_release` on host threads. One waits on the gate word alone;
the other drives it against the real `platform_retire_aicore_group`, where the
worker stays held until the whole group has acknowledged and the production
retire path publishes the gate. The third covers HBG resident's autonomous
startup failure: with a stale previous-run RELEASE, it waits for the AICPU's
actual EXIT signal before ACK, then for close before return. All publish the
release before joining and
assert non-fatally, so a failed expectation cannot leave the un-timeouted wait
unjoined.

Each of the original seven was confirmed to discriminate by mutating the source
it tests — closing on acknowledgement, closing an unacknowledged core, retiring
serially with a per-core budget, reporting every core released, dropping the
validation, and bypassing the group from the then-existing single-core entry. Every mutant fails
the case that owns its property, and the unmutated source passes repeatedly. Two
of the seven were rewritten after the first mutation round showed they passed
against a serialized retirement and against that entry closing on
the signal alone. The three gate cases were added with the gate itself and are not
covered by that mutation round.

## Initialization and retirement ownership

An emergency can precede a peer's barrier-free assignment. Reading that
peer's tracker as a stable partition can retire an incomplete set, consume
its claim before its cores initialize, or retire the same cores again as
orphans. Clearing trackers between runs does not synchronize this run's
initialization.

Each core's atomic retirement state has READY and REQUESTED bits. Publishing
READY releases initialization; requesting retirement acquires it.
The operation that observes only the other bit owns retirement. A request
before READY is serviced by the initializer, without requiring it to reach
the dispatch loop or normal shutdown. Barrier-free groups use the handshake's
fixed blocked partition even when tracker assignment fails. Serial startup
publishes assigned groups after its handshake barrier, or gives all cores to
one fallback publisher if assignment cannot complete. Emergency requests every
core before waiting, without reading a peer's partially assigned tracker. A
single retired boolean is insufficient here: skipping an unpublished core loses
a late request, while marking it retired consumes its claim before it is usable.
Pending requests are serviced after publication; no claim primitive guarantees
progress if the initializer itself never reaches publication.

`test_scheduler_retirement.cpp` in the A5 TMR unit directory drives the
production scheduler cold path with an observable retirement sink. Its twelve
cases cover late and partial initialization, concurrent publication and
retirement, normal/emergency exactly-once ownership, serial failure and success,
reset between generations, and fatal publication observed through completion.
The late and partial initialization cases fail against the original claim. It
also checks one emergency batch/deadline across owners, overlapping subsets, and
return-gate reset between generations, and
`ConcurrentOverlappingSetsClaimEachCoreOnce` has five threads retire arbitrary
overlapping, duplicated core sets from one start line and asserts every core is
retired exactly once — the per-core claim, not a per-caller one. Counts by file:
platform group 10, real sim inline wait 3, scheduler 12. Simulation checks
ordering, not physical MMIO completion.

## Still open

The read-back inside the window close has no observable effect on a simulated
register block; its justification remains the memory-attribute argument in
`docs/hardware/mmio-performance.md`. The scheduler tests use DFX disabled and
do not validate PMU finalization or device MMIO behavior on silicon.

On A5 DT device 0, the gated HBG candidate passed the selected empty,
mixed-chain, vector, and explicit legacy scene tests; the selected TMR
alternating and Dense16 cases also passed golden. The six focused C++ test
targets passed. This is normal-path coverage, not a hardware fault injection.
The official TMR benchmark ran each selected case for 50 rounds in one
main/candidate/main sequence on that device. Against the mean of the two main
arms, Effective changed by -10.79% for alternating Case1 and +2.76% for
Dense16. The latter is an unresolved workload-specific regression: the Sched
window includes AICore execution and dependency waits, so its change cannot
be attributed directly to the post-dispatch retirement tail.
