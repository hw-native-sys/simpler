# L2 swimlane: defer per-task `wmb()` from commit fast path to rotation

**Date**: 2026-06-01
**Verdict**: dropped — measured no win above noise on hardware

## Question

`l2_swimlane_aicpu_complete_task` in `src/a2a3/platform/src/aicpu/`
issues one `wmb()` (ARM64 `dsb st`) per AICPU task completion,
immediately after writing `l2_swimlane_buf->count = new_count` and
before checking whether the buffer is full enough to rotate. The
proposal: remove this per-task barrier and only wmb at the publication
point — inside `switch_records_buffer` (full-buffer rotation) and
`l2_swimlane_aicpu_flush` (partial-buffer end-of-run flush). Host can
only observe `count` after AICPU enqueues the buffer onto a per-thread
ready queue, so the per-task wmb has no observer until rotation.

Expected savings on paper: ~1 `dsb st` per completed task per core.
Same shape as the `aicore_rotate` path and all phase-record paths,
which already wmb only at publication.

## What was tried

Patch (3 hunks in
`src/a2a3/platform/src/aicpu/l2_swimlane_collector_aicpu.cpp`):

1. Remove the `wmb()` immediately after `l2_swimlane_buf->count =
   new_count` in `l2_swimlane_aicpu_complete_task`.
2. Add `wmb()` at the top of `switch_records_buffer`, right before
   `enqueue_ready_buffer(...)` (after the empty-free-queue early
   return).
3. Add `wmb()` in `l2_swimlane_aicpu_flush`, right before
   `enqueue_ready_buffer(...)` on the AicpuTask buffer flush path.

Correctness verified on onboard a2a3 (device 4 via `task-submit`):

- `l2_swimlane` STs with `--enable-l2-swimlane --enable-dep-gen`: pass
- `paged_attention_unroll` Case1 at level 4: pass

Benchmark: `paged_attention_unroll` Case1 at swimlane level 4, 16
iterations each direction, sum of per-thread `sched_cost` extracted
from the device debug log
(`~/ascend/log/debug/device-N/device-*.log`, the `log_l2_perf_summary
[V9] "[scheduler_cold_path.cpp:...] Thread N: sched_cost=X.YYYus"`
lines). 4 scheduler threads, 1024 AICPU `complete_task` invocations
per launch (~256 per thread).

## Result

| Condition | mean sum-sched_cost (16 iter) | median |
| --------- | ----------------------------- | ------ |
| Baseline (per-task wmb) | 3574.2 µs | ~3564 µs |
| Patch (wmb at publication) | 3591.0 µs | ~3568 µs |

Per-iteration standard deviation is ~30 µs; iterations 1 and 11 in
each direction show 100–250 µs cold-cache / contention outliers.
Difference between conditions is within the noise envelope and flips
sign depending on whether outliers are trimmed.

Back-of-envelope expected saving: 256 wmbs × ~30 ns ≈ 7.7 µs per
thread, summed across 4 threads ≈ 30 µs. Below the per-iter
variance, so the benchmark cannot discriminate even if the saving is
exactly as predicted.

## Why not (now)

- DFX path prioritizes **accuracy + simplicity over performance**
  (see `.claude/rules/...` and the user feedback this is informed
  by).
- The patch moves a barrier between two locations that are
  semantically equivalent for correctness, so reading the code
  costs the same; the cognitive cost is **proving** that no observer
  reads the buffer between `complete_task` returning and the next
  rotation. That proof is non-trivial when AICore runs concurrently
  on the same shared memory (AICore reads its own pool's
  `current_buf_seq`, not AicpuTaskPool — so the proof holds — but a
  reader has to walk it).
- Without a measurable win the cost-benefit doesn't justify even a
  small risk.

## When to reconsider

Re-open this **only** if a profile shows the per-task wmb as a
material fraction of either:

- `sched_complete_cycle` on a workload with **many short
  (sub-microsecond)** AICPU tasks where the wmb cost stops being
  amortized away by record-write store-buffer latency, OR
- AICPU thread CPU time on a sustained streaming workload (long
  enough that the cold-cache effects are diluted and 16+ iter
  averaging discriminates a single-digit-µs signal).

Test design for the re-investigation should:

1. Use a workload with **≥10k** AICPU complete events per launch so
   the per-iter signal grows past the noise floor.
2. Run the benchmark inside a single `--rounds N` invocation (one
   pytest call) rather than 16 separate launches, to eliminate the
   per-launch outlier tail dominating the variance.
3. Extract `sched_complete_cycle` specifically (not just `sched_cost`)
   from `PTO2_SCHED_PROFILING` builds — that's the bucket the wmb
   contributes to.

## References

- Patch (never merged, lived only on the
  `perf/swimlane-defer-wmb` local branch on 2026-06-01).
- Related: `aicore_rotate` and `acquire_phase_slot` already use the
  publication-point wmb shape, see
  `src/a2a3/platform/src/aicpu/l2_swimlane_collector_aicpu.cpp`.
- DFX priorities memory note (user-level): "DFX path prioritizes
  accuracy and simplicity over performance — do not propose
  micro-optimizations on profiling / swimlane / diagnostics paths".
