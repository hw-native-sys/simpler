# Arbitrary-depth transitive reduction of WAIT edges on the AICPU orchestrator

**Date**: 2026-08-13
**Verdict**: scoped to 1-hop; arbitrary-depth dropped for the device
orchestrator, deferred to any future host-resident graph (hbg)

## Question

Issue #1375 wants redundant ordering edges removed: if `A→B→C` already
orders `A` before `C` transitively, the direct `A→C` WAIT edge carries no
readiness information and can be dropped (its RETAIN half, when present, must
stay — that is the split PR1 built). The obvious generalization is a full
transitive reduction: for every direct edge `P→C`, drop its WAIT flag if `P`
reaches `C` through *any* chain of other producers, at any depth. On a DAG the
transitive reduction is unique and maximal, so this would remove the most
edges.

The intuition that makes a contributor reach for it: "we already walk the
fanin; just keep walking ancestors until we've found every redundant edge."

## What was tried

Sketched the arbitrary-depth pass against the `tensormap_and_ringbuffer`
orchestrator's actual data model (`pto_orchestrator.cpp`,
`pto_ring_buffer.h`):

- The orchestrator runs on the AICPU and builds each consumer's fanin
  **incrementally at submit time** from the tensormap overlap lookup plus
  creator retention. It never materializes the whole task graph — there is no
  adjacency structure to walk, only the current task's `PTO2FaninBuilder` and
  the producers' own `fanin_inline_slot_states[]`.
- A producer is identified by its slot pointer, resolved to `(ring, slot)` via
  `p->ring_id` + pointer subtraction. That identity is only valid **while the
  slot is live**. Ring slots are reused: once a producer is consumed and its
  slot recycled, walking to it as an ancestor reads a different task's state.
- To answer "does `P` reach `C` at depth ≥ 2" you must either (a) keep a
  persistent transitive-closure / ancestor set per live slot — unbounded
  memory that grows with graph width and must be maintained on every edge, on
  a device path that must not allocate — or (b) re-walk producer→producer
  chains that may already be reclaimed, giving wrong answers under slot reuse.

## Result

Arbitrary-depth is unsound on this model without a persistent closure the
device cannot afford. **1-hop is both sound and cheap** because it only ever
inspects the consumer's *own direct producers*, all of which are still live:

- For edge `P→C`, look at each other direct producer `Q` in `C`'s fanin
  builder and ask whether `P` is a `WAIT|RETAIN` fanin of `Q` (the `P→Q→C`
  diamond). `Q`'s fanin stores only a `(ring, slot)` pointer, not a generation,
  so slot reuse could otherwise make `P`'s pointer resolve to an unrelated task
  `C` also depends on. Two conditions together keep `P`'s identity valid:
  - **`Q` is still live** (`task_state < COMPLETED`). `on_task_release(Q)` —
    which drops `Q`'s RETAIN pins — runs at `Q`'s *completion*, not its reclaim
    (`pto_scheduler.h`). A completed `Q`'s fanin pointers may already resolve to
    reused slots, so a completed `Q` is skipped. This loses no useful reduction:
    `Q` completing implies `P` completed (`Q` depends on `P`), so `C`'s `P→C`
    WAIT is already satisfied and clearing it changes no readiness.
  - **`P→Q` carries RETAIN.** A live `Q` therefore still holds `P`'s pin, so `P`
    is not consumed and its slot not rebound. A `WAIT`-only `P→Q` released `P`'s
    pin at `Q`'s wiring and gives no such guarantee even for a live `Q`.

  Reuse of `P`'s slot requires *all* of `P`'s pins released, and `Q` holds one
  until it completes — so a live `Q` with a RETAIN `P→Q` is the exact invariant
  that proves the walked `P` is the same task `Q` recorded.
- Membership ("`P` is one of `C`'s producers") is O(1) via
  `fanin_seen_epoch[ring][slot] == builder.seen_epoch`.
- Only `C`'s fanin is modified — clearing WAIT on a direct edge already
  covered by a two-hop WAIT path is a subset removal, sound on a DAG.
- Cost guard: `reduce_wait_edges` runs only when `2 ≤ count ≤
  PTO2_FANIN_INLINE_CAP`, so spill-heavy fanins skip it entirely. Treating
  `PTO2_FANIN_INLINE_CAP` as a fixed constant, the inner cost is
  `O(count² · fanin(Q))` — a per-candidate scan of the `redundant[]` set
  (`≤ count`) sits inside the `count · fanin(Q)` walk.

This closes acceptance #1 (diamond) and #2 (creator edge demoted to
RETAIN-only, producer alive until the consumer releases even past scope end)
without any graph-wide structure.

## Why not (now)

The device orchestrator's incremental, slot-reusing model has no reliable view
past one hop. Buying arbitrary-depth means a persistent per-slot ancestor set
maintained on every edge — memory and per-task maintenance cost the AICPU
dispatch path cannot carry, to remove edges beyond the first hop that the
diamond case (the one the issue actually motivates) does not need.

## When to reconsider

`host_build_graph` (hbg) builds the entire task graph on the host, to
completion, before any scheduler thread starts (see `dep-gen.md` §2.2). A
host-resident full DAG **can** afford a real transitive reduction, and that is
the natural home for the arbitrary-depth pass. If hbg grows a reduction step,
implement it there against the materialized graph rather than trying to make
the AICPU orchestrator hold closure state. Revisit for tmr only if the
orchestrator ever gains a persistent host-side or AICPU DAG representation.

## References

- Issue #1375 (split WAIT/RETAIN + transitive reduction), acceptance #1/#2.
- PR1 (#1806, merged): WAIT/RETAIN orthogonal-flag infrastructure.
- PR2: `reduce_wait_edges` in
  `src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`.
- Issue #1827: `DepGenRecord` lacks per-dep kind (why explicit-dep replay
  cannot distinguish `add_dep_wait()`), sub-issue of #995.
- `.claude/rules/codestyle.md` §5 (no sleeping / bounded work on the dispatch
  path) — the cost guard exists to keep the pass off the unbounded path.
