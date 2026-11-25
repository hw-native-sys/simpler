# A5 percounter: the rejected attempts on the dependency-successor gap

Attempts to shorten the gap between a task and its dependent successor on A5
`percounter`, all measured on `pypto_lib_dsl_all_sync64/c2a_full`. The first
three were rejected outright; the last two are null results kept or dropped on
structure rather than on their own measurement. They are recorded together
because they share one root cause, and because each one looks obviously correct
until it is measured.

## The gap being attacked

L2 on upstream `b70bfd5bf` puts 13.97 us between `score_topk_mix` and
`sparse_attention_mix`: 1.53 us publishing the predecessor's completion, 0.55 us
propagating the dependency release, 6.65 us for the successor's 32 block claims,
and 5.25 us of admission. Twenty-three tasks with predecessors all show a
positive gap, median 6.0 us, 544 us against a 1584 us wall.

L4 attributes the claim segment to serialization rather than volume. A returning
GM `fetch_add` costs 613 ns against 2.4 ns for the fire-and-forget form, so the
whole cost is the return trip; 32 blocks over four claim sectors is eight serial
ticket layers, 8 x 613 ns = 4.9 us, which matches the 4.67 us of same-address
serialization L2 resolves inside that 6.65 us.

## Rejected: a wider pending window, with and without a lookahead draw

`PERCOUNTER_PENDING_CAPACITY` is 1, so a lane claims only when it holds nothing.
Its claim therefore always sits between two consecutive tasks with nothing to
overlap it. Raising the capacity to 2 gives the lane a free position, and gating
the top-of-step claim on `(occupied & (occupied - 1)) == 0` instead of
`occupied == 0` lets the pass that selects a task also draw its successor, so
the draw and the arrival it leads to run under the current kernel.

Both halves are measured against the same baseline, six rotated pairs, L0, one
warmup and ten rounds:

| Arm | Mean (ms) | Change | Pairs |
| --- | --------: | -----: | ----- |
| capacity 1 (baseline) | 1.4741 | — | — |
| capacity 2, no lookahead | 1.4867 | +0.86% | 5/6 |
| capacity 2 + lookahead | 1.4739 | +1.60% | 6/6 |

The lookahead arm is paired against its own interleaved baseline runs, which
averaged 1.4507 ms; the other two share the 1.4741 ms column above.

Widening the window alone costs 0.86%, and the lookahead costs another 0.75%.
The mechanism is a scale mismatch that L4 states directly: `percounter_step`
totals 133,244 us, of which 64,209 us is the nested `run_task`, leaving 1.41 us
of scheduling work per pass across 48,857 passes. Slot scanning, record service
and record discovery all run every pass and all scale with the capacity, so
doubling it spends tens of thousands of microseconds to save a claim path that
totals 4,058 us. No gating refinement recovers that; the window is the wrong
place to buy overlap.

The lookahead also changes acquisition in a way worth recording even though it
was reverted. `percounter_claim` short-circuits on `service_pending` when the
lane already holds work, and that path advances `source.segment`. Drawing ahead
therefore consumes the lane's forward cursor while it is busy. The advance is
safe -- a draw fails only once every sector reports its tickets issued -- but it
makes a busy lane skip segments a free lane would have examined.

## Rejected: removing the dependency-probe throttle

`percounter_step` probes fanin on every pass for the first 16 idle passes and
every fourth pass after that, so a lane waiting on a long predecessor can take
up to four passes to notice the release. Against a 1.41 us pass that is the
right order of magnitude for the 5.25 us admission segment, and the probe had
just become much cheaper: the sync readiness check moved from four GM arrival
sectors to one SRAM CntNotify read.

Probing on every pass measures +0.29%, 3/6 pairs, t = 0.79 against a paired SD
of 0.90 pp. Indistinguishable from the baseline, so the throttle is not what the
admission segment is waiting on.

That negative is what identifies the segment. Admission is a sync cohort
assembling, and a cohort cannot start until its last block arrives; each block
arrives after its own claim. The admission segment is therefore not an
independent cost but the shadow of the claim segment's spread, which is why
making discovery faster changes nothing and why the claim serialization is the
only term worth attacking.

## What the three results leave

Per-pass cost is the product, not an overhead to be traded against: anything
that makes an idle pass more expensive loses more than any claim-path saving can
return. Concurrency on the claim path is not the lever either -- an uncontended
returning `fetch_add` still pays the full 613 ns round trip, which is why
widening the claim sectors was already measured worse (wide-task 32 sectors
465.096 us, global 8 sectors 452.730 us, against a ~447 us baseline).

The term that survives is the returning atomic itself. Removing it for sync
boundaries sized to their owner count -- block `rank` to the owner of rank
`rank`, no ticket drawn -- measures -2.05% on the same six rotated pairs, 6/6,
t = -7.35, and -0.722% across the 12-case suite sum, 8 of 12 improving. The
gate is exact equality, so a cohort one block narrower than its domain's owner
count keeps the ticket path; the four cases that move are the ones whose sync
cohorts are full-width.

## Rejected: relaxing the static gate below exact width

The gate is exact equality, which leaves every cohort narrower than its
domain's owner count on the ticket path. Relaxing it to `blocks <= owners`
looks free: an owner whose rank is below the block count takes that block, a
higher rank finds nothing and skips, and no two owners can name the same block
either way.

It costs 2.02%, 6 of 6 pairs worse, paired SD 0.36 pp, t = 13.8 -- the largest
and cleanest negative of the series.

What the ticket path buys is placement, and only a full-width cohort has
nothing to lose by giving it up. At exact width every owner takes exactly one
block, so the ticket merely tells each owner the block it was going to get;
below that width the tickets go to whichever owners reach the boundary first,
and the cohort waits on its fastest free owners. Binding block `rank` to owner
`rank` instead makes it wait on those particular owners however busy they are,
and a cohort starts only when its last block arrives. The narrower the cohort,
the more owners are idle while it waits for a named one.

So the two paths are not a fast case and a slow case to be widened toward each
other. They answer different questions, and the equality is what separates the
cohorts that need an answer from the ones that do not.

## The closure bitmask is kept on structure, not on its own measurement

Turning `closed_sectors` from a count into a bitmask with a publication flag
removes the second returning `fetch_add` a sector's last drawer used to make to
elect the segment's single closer. On its own that is worth 0.57 us of a 6.65 us
claim segment, and six rotated pairs cannot see it: -0.158%, paired SD 0.90 pp,
t = -0.43, against a baseline arm whose own range is 2.1%. Resolving an effect
that size needs roughly 200 pairs, so the single-case A/B is the wrong
instrument and no amount of rerunning it will decide the question.

It is kept because the static assignment above depends on it. A statically
assigned owner publishes its sector's bit with no returned value; under the
count representation that same owner would have to run the electing
`fetch_add`, putting a 613 ns round trip back on four of the cohort's owners --
including, by construction, owners on the cohort's critical path. The bitmask
is what makes the sector closure free, so reverting it would cost most of what
the static path just bought.

## Rejected: deriving the CntNotify addresses from a base and a stride

A lane caches its cohort's notification slot, so the table costs two device
loads per cohort it visits rather than two per probe. Removing even those meant
computing the pair instead: have the Host check that its lease came out
consecutive and publish one base pair and a stride.

It never fires. The pool hands out registers in descending pages -- 0x20 apart
within a 4 KiB page, sixteen to a page, then a jump of several pages -- so for
the 27 cohorts of this case the deltas run `-4064` sixteen times and then
`-147424`, `-4576`, `-143328`, `-184288`. The addresses are not an arithmetic
sequence and no base-and-stride form describes them.

The experiment is worth recording for how it nearly passed as a result. The
derivation is published by the Host and consumed on the device with no visible
failure mode when it does not apply, so its first A/B measured two identical
arms and returned the null one would expect from a real change with no effect.
The observability that settled it had to be added on purpose, and had to be
`LOG_WARN`: the benchmark runner pins the host log threshold at `TIMING`
([historical runner.cpp](https://github.com/hw-native-sys/simpler/blob/1ea06c2aee35993c288887582cb7f802b2724546/tests/fdwic/dsv41_scheduler/production/runner.cpp)), so a `LOG_INFO` line
never reaches the captured host log. **A gated optimization needs a published
signal saying whether its gate opened, before its A/B means anything.**

## Null: one address per cohort instead of a pair

The same probe showed the second word was redundant: a slot's ADD alias is its
SET address plus 8, which is the rule the Host itself applies, so the device
can apply it too. The uploaded table becomes one word per cohort and the slot
costs one load.

Six rotated pairs measure +0.21%, t ≈ 0.2 -- indistinguishable, and this round
the paired SD was 3.3 pp rather than the 0.36-0.90 pp of the earlier rounds,
with seven of the box's eight devices held by other users at the time. The
effect being sought is about 27 loads per lane out of ~62,000, a ratio of load
counts and not of wall time: converting it would need a per-load latency and a
wall figure this experiment did not measure. Six pairs at a 3.3 pp paired SD
cannot resolve an effect of that order either way, so the result is
inconclusive rather than a measured null, and a larger sample would narrow it.

It is kept on structure rather than on that measurement: strictly fewer loads,
half the uploaded bytes, and one wire field where there were two. There is no
configuration in which the second word carries information the first does not.

## What the branch measures end to end

The increments above are each measured against the arm before them, and two
such arms built from the same source measured 17.4519 and 17.4032 ms on
different days -- a 0.55% drift larger than the first increment itself. So the
increments are not summable. The branch is measured against upstream directly,
and in the same session as the arm carrying the publication-floor defect below,
so these three figures may be subtracted from one another:

| Arm | 12-case sum (ms) | vs base | Cases improved |
| --- | ---------------: | ------: | -------------- |
| `c22eaecd6` (base) | 17.476830 | — | — |
| with the defect | 17.286383 | **-1.090%** | 11 of 12 |
| as merged | 17.252801 | **-1.282%** | 10 of 12 |

| Case | Base (ms) | Merged (ms) | Change |
| ---- | --------: | ----------: | -----: |
| `pypto_lib_latest/engram` | 0.826260 | 0.789891 | -4.40% |
| `pypto_lib_dsl_all_sync64/c2a_full` | 1.493030 | 1.437645 | -3.71% |
| `pypto_lib_dsl/c2a_full` | 1.465662 | 1.429940 | -2.44% |
| `pypto_lib_latest/c1a_reindex` | 0.218942 | 0.214123 | -2.20% |
| `pypto_lib_dsl/moe` | 1.244621 | 1.224962 | -1.58% |
| `pypto_lib_latest/c2a_full` | 0.222299 | 0.220079 | -1.00% |
| `pypto_lib_dsl/layers_19_21` | 8.005288 | 7.941562 | -0.80% |
| `pypto_lib_latest/layers_19_21` | 2.199451 | 2.189029 | -0.47% |
| `pypto_lib_latest/c1a_reuse` | 0.174590 | 0.173935 | -0.38% |
| `pypto_lib/c2a_full` | 0.936895 | 0.936677 | -0.02% |
| `pypto_lib_latest/swa` | 0.164321 | 0.164739 | +0.25% |
| `pypto_lib_latest/moe` | 0.525470 | 0.530220 | +0.90% |

Two passes with the arm order rotated, one warmup and ten measured rounds per
case, all 72 runs clean. Cards are named explicitly -- device 6, or 6 and 7 for
the two-rank cases -- rather than taken with `--device auto`, which hands out
the lowest free index and had put 23 of 29 runs of an earlier attempt on device
0. Pinning every arm to one card removes card-to-card variation from the
comparison; it did not reduce the paired spread, which stayed at 1.60 pp, so
that spread is the benchmark's own and not contention between cards.

The largest absolute contribution is not the largest relative one:
`pypto_lib_dsl/layers_19_21` moves 0.80% but carries 8.0 ms of the suite's
17.5 ms, so it alone accounts for 0.064 ms of the 0.224 ms total.

## The correctness defect did not account for the gain

Review found that the static assignment left `percounter_source_floor`
releasing a boundary on sector exhaustion, which on a statically assigned
boundary means only that the owners of the highest blocks have claimed. A lane
could therefore advance its floor past a boundary still holding the block
reserved for it, enter a younger cohort ahead of work it still owed, and
publish that floor to its peers; a MIX coordinator could reach `mix_closed`
before publishing its own reserved block, after which the cohort's arrivals
could never complete.

The natural assumption is that an overly permissive gate is part of why the
arm was fast, and that correcting it costs most of the gain. It does not.
Holding the floor until this lane's own claim cursor passes measures
**-0.194%** against the defective arm -- a further improvement, not a cost,
with a per-case mean of -0.02% at a 0.31 pp spread. The reason is that the two
effects are not the same size: the corrected floor reads no GM word at all on a
static boundary where it previously read one per call, and it is consulted up
to twice per pass, while the premature release only matters inside the window
where a lane actually still owes a block. Removing a read from every pass beats
widening a rare window.

That is worth recording because the opposite is the usual case, and because the
arithmetic only exists at all thanks to measuring both arms in one session: the
0.55% cross-day drift above is larger than the entire effect being attributed.
