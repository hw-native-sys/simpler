# Reordering the scheduler loop's graph control pass ahead of the `sync_start` drain check

**Date**: 2026-09-20
**Verdict**: dropped — superseded by releasing a Graph shell's roots on the
completion path. The reorder fixes the hang, but keeps the invariant it relies
on maintained by code placement alone.

## Question

Issue [#2256](https://github.com/hw-native-sys/simpler/issues/2256): a
`require_sync_start` body root deadlocks against an ordinary sibling root staged
by the same shell release. The sibling holds gated cores until its own doorbell
rings; the cohort cannot fit, enters the global drain, and the drain waits for
cores that only `activate_graph_task -> graph_route_ready_roots ->
push_ready_routed` can release — a call that sat behind the drain check's
`continue` in `resolve_and_dispatch`.

The direct fix is to move the scheduler loop's graph control block ahead of that
check. It is a block move of ~26 lines with no statement added or removed, its
own comment already stated the argument ("Graph control work never consumes an
AICore"), and it demonstrably clears the hang. Anyone meeting this deadlock will
reach for it first.

## What was tried

- The reorder itself, in [#2284](https://github.com/hw-native-sys/simpler/pull/2284):
  graph control moved ahead of the drain check on both a2a3 and a5. Verified
  with a new scene (`tests/st/a2a3/host_build_graph/graph_sync_start_sibling_root/`)
  that hangs against the unfixed runtime and passes with the change, plus the
  `host_build_graph` sweeps on `a2a3sim` and `a5sim`.
- Archaeology on why the order was what it was. `git log -S` puts the drain
  check in the repository's import commit, in the same position the
  `tensormap_and_ringbuffer` loop still has it: immediately after Phase 1, the
  only step that frees cores. Before graph execution existed, the code between
  that check and dispatch was empty — the pre-#1444 source says so in a sentence
  #1444 deleted, *"The scheduler loop goes straight from completion detection to
  dispatch"* — so "skip everything after the check" was exactly "skip dispatch",
  which is what `SyncStartDrainState`'s contract says the drain does. #1444
  inserted the graph control block into that gap under the only constraint it
  cared about ("keep this ahead of dummy/regular dispatch") and the equivalence
  broke silently.
- The cost of the reorder on the drain's critical path. The ack barrier makes
  anything a thread does before reaching `handle_drain_mode` into cohort latency,
  and the insufficient-resources retry repeats it every round. Measured against
  the code rather than a profile: the graph queues are MPMC, so per retry at most
  one thread pays a materialization slice (`GRAPH_MATERIALIZE_SLICE_TASKS` = 4)
  and the others one empty pop; activation is once per graph, not once per retry,
  because `route_cursor` is monotonic and the shell is popped once.

## Result

The reorder works, and its added drain latency is bounded and small. What it does
not do is remove the class of defect. It leaves this invariant in force:

> every path that can free a core another task is waiting on must sit before the
> drain check

and that invariant is held by nothing but the order of two blocks in one
function, plus a comment. No compile-time signal, no test signal. The same
livelock returns the next time a control pass is added below the check, and it
returns as a zero-diagnostic hang: `SIMPLER_SCHEDULER_TIMEOUT_MS` never fires,
because the drain spins live outside the dispatch loop's wall-clock budget and
the budget check sits after the `continue`.

The asymmetry underneath is what makes that likely rather than hypothetical. A
shell's body roots are *staged* through the early-dispatch publish chain, exactly
like any other candidate, but were *released* through a scheduler-loop queue,
while every other early-dispatch release happens inline on the completion path —
on the resolution thread, which never takes part in a drain. One Graph, two
mechanisms.

## Why not (now)

The shipped fix removes the asymmetry instead of accommodating it:
`push_ready_routed`'s `TaskKind::GRAPH` branch calls `activate_graph_task`
inline. Both halves of a Graph's early dispatch then use the machinery a
top-level task uses, the scheduler loop stops being a release path at all, the
drain's critical path gains nothing, and the invariant above holds structurally
rather than positionally.

Its cost is real but narrow: routing — and the doorbell MMIO writes for a staged
root's cores — moves onto the single resolution thread. That is the same thread
where an ordinary candidate's release already rings, so the work is concentrated,
not created.

A lighter variant was also rejected: move only the activation half ahead of the
check and leave materialization behind it. It fixes the hang with the smallest
steady-state cost of the three, but preserves the positional invariant, which is
the property this decision is about.

## When to reconsider

If the resolution thread's completion path becomes a measured bottleneck for a
Graph with many pre-staged roots — the doorbell loop is one MMIO write per staged
core — routing could be handed back to a core-owning thread. The drain-ordering
constraint returns with it, and would then need an explicit guard rather than a
placement convention: a `sync_start_pending` check in the pass itself.

## References

- Issue #2256; PR #2284 (the reorder, not merged).
- [#2167](https://github.com/hw-native-sys/simpler/pull/2167) introduced
  `stage_graph_roots_early`, i.e. a shell release staging the body's roots.
  [#1444](https://github.com/hw-native-sys/simpler/pull/1444) added the graph
  control block to the scheduler loop.
- [2026-07-sync-start-drain-retry-aba.md](2026-07-sync-start-drain-retry-aba.md)
  for the drain protocol this interacts with.
- `src/common/host_build_graph/docs/GRAPH_EXECUTION.md` records the resulting
  contract.
