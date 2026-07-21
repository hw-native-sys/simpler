# Cross-task batched publish: hoist wmb across distinct tasks in one pop

**Date**: 2026-06-06 (initial), 2026-06-08 (revised)
**Verdict**: **shipped with sync_start exclusion** — cross-task batched publish fires for any pop whose batch carries no `requires_sync_start()` task; pops that contain a sync_start task fall back to the per-task wmb path. 10/10 `spmd_sync_start_stress` runs PASS under the gated design; qwen3 decode_layer recovers ~60 ns first-to-last AICore start (vs ~6 µs in the per-claim-only design).

## Question

The batched-publish optimization on `tensormap_and_ringbuffer`'s
`SchedulerContext::dispatch_shape` collapses N store-store fences into
one **per claim of one task** — claim = up to `MAX_CLUSTERS` blocks all
belonging to the same `slot_state`. For SPMD tasks with `block_num > 1`
the win is real and shipped in this PR; for workloads where every
"task" has `block_num == 1` (decode-style small kernels, single-block
matmul tiles), each task still pays its own wmb because the per-task
for-bi loop publishes between iterations.

The proposal: hoist the `handles[]` array, the `wmb()`, and the publish
loop out of the per-task body and run them **once per `pop_ready_tasks_batch`
call**, so the wmb amortizes across the distinct tasks the scheduler
just popped (up to `cores.count() ≤ MAX_CLUSTERS` of them).

## What was tried

Patch in `scheduler_dispatch.cpp::dispatch_shape`:

- Lifted `PublishHandle handles[CoreTracker::MAX_CLUSTERS * 3]` and
  `int handle_count` to the outer `while (cores.has_value() && !entered_drain)`
  scope (above the `for (bi)` loop).
- Replaced the per-task `wmb() + publish loop` with an accumulation:
  each task's `prepare_block_for_dispatch` appends to the shared
  `handles[]`. A `flush_publish` lambda (wmb + publish loop + reset
  `handle_count` + `made_progress = true`) runs at the end of the
  while-iteration.
- In the sync_start drain branch, called `flush_publish()` BEFORE
  `enter_drain_mode()` so the prepared-but-unpublished cores are
  MMIO-visible before drain starts walking trackers.

Size argument: sum of claims across distinct tasks in one pop is
bounded by `cores.count() ≤ MAX_CLUSTERS`, and each block contributes
≤ 3 subtasks for MIX, so the existing `MAX_CLUSTERS * 3` array suffices.

## Result

Benchmark (`tools/benchmark_rounds.sh` 100 rounds, device 4 against
merge-base in a worktree venv): all 9 examples within ±2.78% Total on
the default suite — workloads are serial chains where
`pop_ready_tasks_batch` returns 1 task at a time, so the cross-task
path never engages. `benchmark_bgemm` was the only one to move
materially (−2.78% Total) because it dispatches independent matmul
tiles that occasionally co-arrive in the same pop.

**Correctness — `spmd_sync_start_stress` regression**:

Local repro on a2a3 onboard (device 4, dedicated lock per run):

| Branch | sync_start_stress pass rate |
| ------ | --------------------------- |
| `upstream/main` (no batching) | 5 / 5 |
| Per-task batched publish (this PR's shipped scope) | 9 / 10 (one runner-flake) |
| **Per-task + cross-task batched publish** | **2 / 5** |

The failing runs all hit `aclrtSynchronizeStreamWithTimeout (AICPU)
failed: 507018`. The stress test submits 54 tasks per round across 6
rounds with a mix of normal MIX, sync_start MIX (`block_num = 12`),
sync_start AIV (`block_num = 8`), and normal AIV — exactly the path
the cross-task change extends through.

The most plausible mechanism: cross-task batching delays MMIO publish
of tasks 0..bi−1 until `flush_publish()` runs immediately before
`enter_drain_mode()`. In the per-task version, each of tasks 0..bi−1
was published with its own wmb between them, so AICore had several
microseconds of head start before drain triggered and the drain
coordinator's `count_global_available()` check could see cores freeing
up. With cross-task batching, AICore receives the bursted MMIO writes
right before drain entry — `count_global_available()` sees those cores
as occupied for longer, the drain elected-worker's
`available < block_num` path triggers more retries, and the
retry / handle_drain_mode cycle eventually trips the
1-second `PLATFORM_OP_EXECUTE_TIMEOUT_US` and stream times out.

The exact race window was not pinpointed — the drain retry path is
designed to be robust to insufficient resources, but combining it
with cross-task batching shifted timing enough to lose. Reverting the
cross-task hoist (keeping per-task batched publish) immediately
restored stability.

## Revised design (2026-06-08) — sync_start exclusion

The follow-up #2 ("skip the cross-task accumulation when any task in
the just-popped batch carries `requires_sync_start()`") shipped. The
detection cost is one `requires_sync_start()` check per popped task,
which is a single mask bit read on memory the scheduler just touched.

```cpp
bool any_sync_start = false;
for (int bi = 0; bi < got; bi++) {
    if (batch[bi]->active_mask.requires_sync_start()) { any_sync_start = true; break; }
}
// ... per-bi loop accumulates handles[]; if (any_sync_start) flush_publish() inside
// the loop; otherwise one flush_publish() at the end of the pop.
```

The drain entry path still calls `flush_publish()` before
`enter_drain_mode()` (same as the v3 attempt) so any prior tasks in
the batch get their MMIO writes out — but when `any_sync_start == true`
the per-task `flush_publish()` already published them, so the drain
flush is a no-op and the head-start between tasks is preserved.

## Measurement under the revised design

a2a3 onboard, dedicated device lock per run:

| Test | Result |
| ---- | ------ |
| `spmd_sync_start_stress` × 10 | **10 / 10 PASS** |
| qwen3 decode_layer (level 4 swimlane) × 8 — per-thread first-wave dt span | median **0 µs**, max 2.72 µs |
| qwen3 decode_layer × 8 — per-thread first-wave st span | median **~60 ns**, max 2.86 µs |
| qwen3 decode_layer × 8 — cross-thread first-dispatch stagger (with the overflow gate from this PR) | median **1.92 µs**, min 0.16 µs |
| qwen3 decode_layer × 8 — wall | median 902.9 µs (within run-to-run noise of the per-claim-only design) |

The qwen3 metrics validate the workload class the doc's original
"When to reconsider" gate identified: per-pop ready count routinely ≥ 8
(50 single-AIC `out_proj` tasks ready together) and kernel duration
short enough (~22 µs each) that the per-task wmb cost matters.

## Root cause found (2026-06-16) — drain ack-barrier had no `completed_` escape

The "exact race window was not pinpointed" note above is now resolved. The
507018 was **not** fundamentally about batched-publish timing — that change
only widened an existing window. The real defect is a liveness bug in the
sync_start drain protocol (`SchedulerContext::handle_drain_mode`):

the three drain spin-waits (sentinel wait, **ack barrier**, non-elected wait)
require all `active_sched_threads_` to keep participating, but a scheduler
thread leaves the dispatch loop (`SchedulerContext::resolve_and_dispatch`) the
instant `completed_` latches — at the loop-top check or via
`SchedulerContext::handle_orchestrator_exit`. A thread that
enters the ack barrier in the window where its peers are exiting on
`completed_` waits forever for acks that can never arrive. The in-runtime 2 s
watchdog can't catch it (it lives in the dispatch loop's idle path, not inside
the barrier spin), so the hang escalates to the 3 s STARS op-exec timeout →
507018 → device poison → force-reset.

`spmd_sync_start_stress` is the only test that hits it because it is the sole
case that maximizes all four required factors simultaneously: many drain
cycles (24 sync tasks), concurrent normal tasks occupying cores (forces the
drain *retry* state to linger), cross-shape MIX+AIV contention on the single
drain slot, and a long multi-round tail that produces thread skew. `--rounds N`
multiplies the per-run hit probability. Pure sync_start tests dispatch their
drains instantly (resources always available) so the barrier is a sub-µs
transient; siblings with contention (`spmd_starvation`) have too few drain
cycles.

**Fix**: every drain spin now honors `is_completed()` and returns when the run
has latched `completed_`, mirroring the existing escape in
`SchedulerContext::handle_core_transition`. Abandoning a drain under
`completed_` is safe — any
pending sync_start task is either already dispatched (a stale re-popped slot)
or moot under teardown, and `deinit()` resets `drain_state_` before the next
run. Applied to both a2a3 and a5.

## References

- PR #989 (this PR): where per-task batched publish + the sync_start-
  gated cross-task batched publish ship together.
- `spmd_sync_start_stress`
  (`tests/st/a2a3/tensormap_and_ringbuffer/spmd_sync_start_stress/`).
- Issue #545 comment #2: the SPMD dispatch-stagger symptom the PR
  was opened to fix.
