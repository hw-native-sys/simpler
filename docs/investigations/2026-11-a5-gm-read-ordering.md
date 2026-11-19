# A5 GM reads: what orders a pair, measured

**Date:** 2026-11-11 · **Silicon:** a5 · **Why:** percounter's cohort queue
decides on pairs of GM words, and review kept asking which of them could be
stale relative to the other. The question could not be settled by reading
`scheduler_memory.h`, whose own comment turned out to describe something
narrower than it sounded.

## What was run

A one-off scene test under `tensormap_and_ringbuffer` on a5 hardware (not
committed; it exists only to answer this). Two independent tasks share one
tensor, **both taking it as an input** -- TensorMap tracks producers, not
consumers, so a read-read pair carries no dependency and the two run at the
same time on different cores. Each kernel reaches the shared words through
the raw address. A writer publishes `data` then `flag`, both monotone; a
reader reads `flag` then `data` and counts how often `data < flag`, which can
only mean it observed the two writes out of order. A compiler barrier sits
between each pair so the question stays about the hardware. One writer only,
checked by an instance counter -- two writers counting independently would
produce the same inversion for a reason that is not ordering at all.

## What came out

Per run, ~285k sampled pairs, writer publishing in a tight loop:

| reads | writer fenced | same cache line | two cache lines |
| ----- | ------------- | --------------: | --------------: |
| two bypass loads (`ld_dev`) | no | 0 .. 11 | 0 .. 7 |
| two bypass loads | yes | 2 | 2 |
| two bypass loads | no, **reader fenced** | **0** | **0** |
| two atomic reads (`atomicAdd` of zero) | no | **0** | 702 .. 2215 |
| two atomic reads | yes | **0** | 830 .. 1918 |
| two atomic reads | no, **reader fenced** | **0** | **0** |

Twelve runs per row for the fenced rows; eight for the rest.

Three results, each of which contradicted an expectation held at the time:

- **A barrier on the writing lane does nothing**, either way the words are
  read. The 4-of-8 to 1-of-8 drop in the first two rows is not significant
  (Fisher, p≈0.28), and the high-signal atomic rows show no effect at all.
- **Reading atomically fixes a pair that shares a cache line and makes a pair
  that does not far worse.** Both words of a shared line reach the same unit,
  which serializes them; two RMWs to different lines go to different units,
  take longer, and diverge further. So `alignas(64)` between two words a lane
  reads together is a hazard, not only a cure for false sharing.
- **A barrier between the reader's two loads closes every case**, and it
  holds across intervening reads -- the last row was measured with three
  unrelated reads between the fence and the payload, which is the shape
  `percounter_sync_promote` has.

Separately, 440k back-to-back trials of "atomic write, then bypass-load the
same word on the same core" produced **no stale read**, with a positive
control (an atomic read-back of the counter) confirming the loop body
survived the compiler. `scheduler_memory.h` says a bypass load "does not
observe a value an atomic wrote until that write has drained -- the writer's
own core included"; the self-read half of that is not reproducible here. The
cross-core half is, and the table above is it.

A stale value was never seen more than one publication behind, over 26
observed inversions. That bounds nothing -- 26 cases cannot bound a tail --
and the one place the code would otherwise lean on it has a barrier instead.

## What it changed

`percounter_observe_fence` and `percounter_gate_read` in the lane loop, and
the fence placements in `percounter_sync_take` and `percounter_sync_promote`.
A barrier that the releasing lane used to carry between its two stores was
removed, because it was measured to do nothing; **if that looks wrong to a
later reader, this table is why it went.**

## What was not measured

Whether two *atomic* writes to different words reach a reader in the order
one lane issued them. The reader side is fenced instead, which is what the
last row covers. SSBUF was not measured at all -- it is on-chip, and nothing
here transfers to it.
