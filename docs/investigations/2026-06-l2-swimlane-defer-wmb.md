# L2 swimlane: defer per-task `wmb()` to buffer publication

**Date**: 2026-06-01 (investigated), 2026-07-08 (landed)
**Verdict**: **landed** — the per-task `wmb()` in `l2_swimlane_aicpu_complete_task`
is removed. Correctness verified onboard a2a3. The speed benefit is below the
hardware noise floor; the change is kept for **uniformity**, not measured speed.

## Change

`l2_swimlane_aicpu_complete_task` used to issue one `wmb()` (ARM64 `dsb st`) per
AICPU task record, right after `l2_swimlane_buf->count = new_count`. It was the
only record-writer on the AICPU collector still doing a per-record barrier; the
phase-record and `aicore_rotate` paths already defer their barrier to
publication. The per-record `wmb()` is removed; nothing is added.

## Correctness

The host reads an `AicpuTask` buffer only after the AICPU publishes it to the
per-thread ready queue. Both publication paths — full-buffer rotation
(`switch_records_buffer` → `L2SwimlaneTaskEngine::switch_buffer`) and end-of-run
flush (`l2_swimlane_aicpu_flush`) — go through
`L2SwimlaneTaskEngine::enqueue_ready` (`profiler_device_engine.h`), which issues
`wmb()` (`dsb st`) before advancing `queue_tails`. A `dsb st` drains every prior
store on that thread, so this single barrier covers the whole buffer's record +
`count` stores before the tail the host polls (with `rmb()`).

The `AicpuTask` buffer has one producer (the scheduler thread that owns the
core) and one consumer (the host, post-enqueue). The AICore never reads it — it
owns a separate pool keyed on its own `current_buf_seq`. So there is no
concurrent observer between `complete_task` returning and publication.

Because the collector was unified into
`src/common/platform/shared/aicpu/l2_swimlane_collector_aicpu.cpp` (#1262) with
publication funneled through `enqueue_ready`'s barrier, the landed change is a
single-line removal — unlike the 2026-06 exploration, which predated that engine
and had to add `wmb()` at each publication site.

## Onboard validation

a2a3 via `task-submit`:

- 2026-07-08: `tests/st/a2a3/tensormap_and_ringbuffer/dfx/l2_swimlane` at
  `@scene_test(level=2)` (the level where `complete_task` runs and writes the
  AICPU task record) — 2 passed. The DFX suite validates record-count
  reconciliation (no silent loss), the failure mode a missing barrier produces.
- 2026-06-01: `l2_swimlane` STs (`--enable-l2-swimlane --enable-dep-gen`) and
  `paged_attention_unroll` Case1 at level 4 — pass.

## Measured benefit: below noise

`paged_attention_unroll` Case1 at swimlane level 4, 16 iterations each
direction, sum of per-thread `sched_cost` from the device debug log
(`log_l2_perf_summary` `Thread N: sched_cost=X.YYYus`). 4 scheduler threads,
1024 `complete_task` invocations per launch (~256 per thread).

| Condition | mean sum-sched_cost (16 iter) | median |
| --------- | ----------------------------- | ------ |
| Baseline (per-task wmb) | 3574.2 µs | ~3564 µs |
| Patch (wmb at publication) | 3591.0 µs | ~3568 µs |

Per-iteration standard deviation is ~30 µs, with 100–250 µs cold-cache /
contention outliers on iterations 1 and 11. The difference between conditions is
within the noise envelope and flips sign under outlier trimming. Back-of-envelope
expected saving: 256 wmbs × ~30 ns ≈ 7.7 µs/thread, ≈ 30 µs across 4 threads —
below the per-iter variance, so the benchmark cannot discriminate it.

The AICPU per-record `dsb st` is the **cheaper** of swimlane's two per-record
barriers. The load-bearing cost is the AICore-side per-record record write-back,
which is **not** removable: AICore cache is non-coherent with the consumer, so
each AICore record must be pushed out before its buffer is handed over.

## Risk carried forward

A sibling barrier change (hoisting `wmb` across distinct tasks in one pop) caused
a `spmd_sync_start_stress` 507018 drain-barrier hang — see
[`2026-06-cross-task-batched-publish.md`](2026-06-cross-task-batched-publish.md).
This deferral is narrower (a single per-record barrier whose only observer
publishes through its own barrier) and is onboard-validated above.

## To measure a real signal (future)

If a profile ever needs to attribute this barrier's cost, size the workload so
the signal clears the noise floor:

1. ≥10k AICPU complete events per launch (short, sub-microsecond tasks, where the
   `dsb st` stops being amortized by record-write store-buffer latency).
2. A single `--rounds N` invocation rather than N separate launches, so the
   per-launch outlier tail does not dominate the variance.
3. Extract `sched_complete_cycle` from `SIMPLER_SCHED_PROFILING` builds — the bucket
   the barrier contributes to — not just `sched_cost`.

## References

- Landed in PR [#1301](https://github.com/hw-native-sys/simpler/pull/1301).
- `aicore_rotate` and `acquire_phase_slot` use the same publication-point barrier
  shape, in `src/common/platform/shared/aicpu/l2_swimlane_collector_aicpu.cpp`.
- DFX path prioritizes accuracy and simplicity over performance; per-record
  barrier micro-optimizations on the swimlane path are otherwise not pursued.
